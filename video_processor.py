import subprocess
from pathlib import Path
import uuid
import logging
import time
import json
import os
import whisper
from transformers import pipeline
import hashlib
import concurrent.futures
import threading

def format_timestamp(seconds):
    """Format seconds into SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def split_text_for_tiktok(text, max_chars=25):
    """Split text into shorter segments for TikTok-style rapid captions"""
    words = text.strip().split()
    segments = []
    current_segment = ""
    
    for word in words:
        if len(current_segment + " " + word) <= max_chars:
            if current_segment:
                current_segment += " " + word
            else:
                current_segment = word
        else:
            if current_segment:
                segments.append(current_segment)
            current_segment = word
    
    if current_segment:
        segments.append(current_segment)
    
    return segments

def write_srt_tiktok_style(segments, file, speed_factor=1.0):
    """Write segments to SRT file with TikTok-style short captions and adjusted timing"""
    srt_index = 1
    
    for segment in segments:
        original_text = segment["text"].strip()
        original_start = segment["start"] / speed_factor  # Adjust for speed change
        original_end = segment["end"] / speed_factor      # Adjust for speed change
        original_duration = original_end - original_start
        
        # Split text into shorter segments
        text_segments = split_text_for_tiktok(original_text, max_chars=25)
        
        if not text_segments:
            continue
            
        # Distribute time evenly across text segments
        time_per_segment = original_duration / len(text_segments)
        
        for i, text_seg in enumerate(text_segments):
            start_time = original_start + (i * time_per_segment)
            end_time = original_start + ((i + 1) * time_per_segment)
            
            start_formatted = format_timestamp(start_time)
            end_formatted = format_timestamp(end_time)
            
            file.write(f"{srt_index}\n{start_formatted} --> {end_formatted}\n{text_seg}\n\n")
            srt_index += 1

class VideoProcessor:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self._tokenizers_parallelism_disabled = False
        self._tokenizers_lock = threading.RLock()  # reentrant lock for safety
        self._progress_lock = threading.Lock()
        self._thread_local = threading.local()  # for thread-specific data
        self._load_models()
        
    def _update_progress(self, step, progress, metadata=None):
        """Thread-safe progress updates"""
        thread_id = threading.get_ident()
        with self._progress_lock:
            if self.progress_callback:
                self.progress_callback(step, progress, metadata)
            logging.info(f"[Thread-{thread_id}] Progress: {step} - {progress}%")
            print(f"DEBUG: [Thread-{thread_id}] {step} - {progress}%")
            if metadata:
                print(f"DEBUG: [Thread-{thread_id}] Metadata: {metadata}")
    
    def _load_models(self):
        """Load AI models in parallel"""
        print("DEBUG: Starting parallel model loading...")
        self._update_progress("Loading models", 5)
        
        def load_emotion_model():
            return pipeline(
                "text-classification",
                model="tasinhoque/text-classification-goemotions",
                top_k=None,
                truncation=True,
            )
        
        def load_whisper_model():
            return whisper.load_model("medium")
        
        # Load models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            emotion_future = executor.submit(load_emotion_model)
            whisper_future = executor.submit(load_whisper_model)
            
            self._update_progress("Loading emotion analysis model", 10)
            self.emotion = emotion_future.result()
            print("DEBUG: Emotion model loaded")
            
            self._update_progress("Loading speech recognition model", 15)
            self.model = whisper_future.result()
            print("DEBUG: Whisper model loaded")
    
    def _disable_tokenizers_parallelism(self):
        """Thread-safe disable tokenizers parallelism"""
        thread_id = threading.get_ident()
        with self._tokenizers_lock:
            if not self._tokenizers_parallelism_disabled:
                print(f"DEBUG: [Thread-{thread_id}] Acquiring tokenizers lock...")
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                self._tokenizers_parallelism_disabled = True
                print(f"DEBUG: [Thread-{thread_id}] Disabled tokenizers parallelism to prevent subprocess deadlocks")
            else:
                print(f"DEBUG: [Thread-{thread_id}] Tokenizers parallelism already disabled")
    
    def run(self, cmd, timeout=150):  # 5 minute default timeout
        """Thread-safe subprocess command execution with timeout"""
        thread_id = threading.get_ident()
        
        # Thread-safe tokenizers disabling
        self._disable_tokenizers_parallelism()
        
        # Convert all arguments to strings
        cmd = [str(arg) for arg in cmd]
        
        print(f"DEBUG: [Thread-{thread_id}] Running command: {' '.join(cmd)}")
        print(f"DEBUG: [Thread-{thread_id}] Timeout set to: {timeout}s")
        start_time = time.time()
        
        try:
            # Add timeout and more detailed process monitoring
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Command succeeded in {duration:.2f}s")
            if result.stdout:
                print(f"DEBUG: [Thread-{thread_id}] STDOUT: {result.stdout}")
            return result
        
        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Command TIMED OUT after {duration:.2f}s (timeout: {timeout}s)")
            print(f"DEBUG: [Thread-{thread_id}] Command: {' '.join(cmd)}")
            print(f"DEBUG: [Thread-{thread_id}] Partial STDOUT: {e.stdout}")
            print(f"DEBUG: [Thread-{thread_id}] Partial STDERR: {e.stderr}")
            logging.error(f"[Thread-{thread_id}] Command timed out after {timeout}s: {' '.join(cmd)}")
            raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd[:3])}...")
        
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Command FAILED after {duration:.2f}s: {' '.join(cmd)}")
            print(f"DEBUG: [Thread-{thread_id}] Return code: {e.returncode}")
            print(f"DEBUG: [Thread-{thread_id}] STDERR: {e.stderr}")
            print(f"DEBUG: [Thread-{thread_id}] STDOUT: {e.stdout}")
            logging.error(f"[Thread-{thread_id}] Command failed: {' '.join(cmd)}")
            logging.error(f"[Thread-{thread_id}] Error: {e.stderr}")
            raise
    
    def transcribe(self, video_path):
        """Transcribe video to get segments"""
        print(f"DEBUG: Starting transcription of {video_path}")
        self._update_progress("Transcribing audio", 25)
        result = self.model.transcribe(str(video_path))
        segments = result["segments"]
        
        print(f"DEBUG: Transcription complete. Found {len(segments)} segments")
        for i, seg in enumerate(segments[:3]):  # Show first 3 segments
            print(f"DEBUG: Segment {i}: {seg['start']:.2f}-{seg['end']:.2f}: {seg['text'][:50]}...")
        
        self._update_progress("Transcription complete", 35, {
            'segments_count': len(segments),
            'total_duration': max(seg['end'] for seg in segments) if segments else 0
        })
        
        return segments
    
    def find_hook(self, segments, min_s=3.0, max_s=6.0):
        """Find the best hook segment with batch emotion analysis"""
        print(f"DEBUG: Finding hook in {len(segments)} segments (duration {min_s}-{max_s}s)")
        self._update_progress("Analyzing emotions for hook detection", 40)
        
        # Filter eligible segments first
        eligible_segments = []
        for i, seg in enumerate(segments):
            dur = seg["end"] - seg["start"]
            if min_s <= dur <= max_s:
                eligible_segments.append((i, seg))
        
        if not eligible_segments:
            raise ValueError(f"No segments found between {min_s}s and {max_s}s duration")
        
        # Batch process emotion analysis
        texts = [seg[1]["text"] for seg in eligible_segments]
        print(f"DEBUG: Batch analyzing {len(texts)} segments")
        
        # Process all texts at once - no fallback, let it fail if it fails
        all_scores = self.emotion(texts)
        
        best = None
        analyzed_segments = []
        
        for (i, seg), scores in zip(eligible_segments, all_scores):
            score = sum(
                item["score"]
                for item in scores
                if item["label"] in {"joy", "amusement", "surprise"}
            )
            
            analyzed_segments.append({
                'text': seg["text"],
                'duration': seg["end"] - seg["start"],
                'emotion_score': score,
                'start': seg["start"],
                'end': seg["end"]
            })
            
            if best is None or score > best["score"]:
                best = {"start": seg["start"], "end": seg["end"], "score": score, "text": seg["text"]}
        
        if best is None:
            raise ValueError("No valid hook segment found after emotion analysis")
        
        print(f"DEBUG: Final hook selected: {best['start']:.2f}-{best['end']:.2f} (score: {best['score']:.3f})")
        print(f"DEBUG: Hook text: {best['text']}")
        
        hook_metadata = {
            'hook_segment': {
                'start': best["start"],
                'end': best["end"],
                'text': best.get("text", ""),
                'emotion_score': best["score"]
            },
            'analyzed_segments': analyzed_segments[:5]
        }
        
        self._update_progress("Hook detection complete", 50, hook_metadata)
        return best
    
    def extract_clip(self, src, start, end):
        """Extract clip segment"""
        print(f"DEBUG: Extracting clip from {start:.2f}s to {end:.2f}s")
        self._update_progress("Extracting hook clip", 55)
        
        out = src.with_stem("hook_" + uuid.uuid4().hex)
        print(f"DEBUG: Output file: {out}")
        
        # Use re-encoding to avoid issues with keyframes
        self.run([
            "ffmpeg", "-y", "-i", str(src), 
            "-ss", f"{start}", "-to", f"{end}",
            "-c:v", "libx264", "-c:a", "aac", 
            "-avoid_negative_ts", "make_zero",
            str(out)
        ])
        
        # Verify the output file exists and has content
        if out.exists():
            print(f"DEBUG: Hook clip created successfully: {out.stat().st_size} bytes")
        else:
            print("DEBUG: ERROR - Hook clip was not created!")
        
        self._update_progress("Hook clip extracted", 60)
        return out
    
    def remove_silence(self, src):
        """Remove silence from video with jump cuts and speed up to 1.25x"""
        print(f"DEBUG: Processing video for silence removal with jump cuts and speed adjustment")
        self._update_progress("Removing silence and adjusting speed", 65)
        
        out = src.with_stem(src.stem + "_processed")
        print(f"DEBUG: Output file: {out}")
        
        # First pass: detect both silence and non-speech regions
        print("DEBUG: Detecting silence and non-speech periods...")
        detect_cmd = [
            "ffmpeg", "-i", str(src), "-af", 
            "silencedetect=noise=-20dB:duration=0.2,silencedetect=noise=-30dB:duration=0.1", 
            "-f", "null", "-"
        ]
        
        # Run silence detection
        result = self.run(detect_cmd)
        stderr_output = result.stderr
        
        # Parse silence periods from stderr
        silence_periods = []
        lines = stderr_output.split('\n')
        silence_start = None
        
        for line in lines:
            if 'silence_start:' in line:
                # Get the noise level from the line - with error handling
                try:
                    if 'noise=' in line:
                        noise_level = float(line.split('noise=')[1].split('dB')[0])
                    else:
                        noise_level = -20  # default noise level
                    silence_start = float(line.split('silence_start: ')[1].split()[0])
                    # Store the noise level with the start time
                    silence_start = (silence_start, noise_level)
                except (IndexError, ValueError) as e:
                    print(f"DEBUG: Error parsing silence_start line: {line}")
                    print(f"DEBUG: Error details: {e}")
                    continue
            elif 'silence_end:' in line and silence_start is not None:
                try:
                    silence_end = float(line.split('silence_end: ')[1].split()[0])
                    start_time, noise_level = silence_start
                    # Only remove silences longer than 0.2s for -20dB or 0.1s for -30dB
                    min_duration = 0.2 if noise_level == -20 else 0.1
                    if silence_end - start_time > min_duration:
                        silence_periods.append((start_time, silence_end))
                    silence_start = None
                except (IndexError, ValueError) as e:
                    print(f"DEBUG: Error parsing silence_end line: {line}")
                    print(f"DEBUG: Error details: {e}")
                    silence_start = None
                    continue
        
        # Merge overlapping or very close silence periods
        if silence_periods:
            silence_periods.sort(key=lambda x: x[0])
            merged_periods = []
            current_start, current_end = silence_periods[0]
            
            for start, end in silence_periods[1:]:
                if start - current_end <= 0.1:  # Merge if gaps are less than 0.1s
                    current_end = max(current_end, end)
                else:
                    merged_periods.append((current_start, current_end))
                    current_start, current_end = start, end
            merged_periods.append((current_start, current_end))
            silence_periods = merged_periods
        
        print(f"DEBUG: Found {len(silence_periods)} silence periods to remove")
        
        if silence_periods:
            # Create video segments between silences
            segments = []
            last_end = 0
            
            for silence_start, silence_end in silence_periods:
                if silence_start > last_end:
                    segments.append((last_end, silence_start))
                last_end = silence_end
            
            # Add final segment if there's content after last silence
            # Get video duration first
            duration_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json", 
                "-show_entries", "format=duration", str(src)
            ]
            duration_result = self.run(duration_cmd)
            duration_info = json.loads(duration_result.stdout)
            total_duration = float(duration_info["format"]["duration"])
            
            if last_end < total_duration:
                segments.append((last_end, total_duration))
            
            print(f"DEBUG: Created {len(segments)} video segments")
            
            if len(segments) > 1:
                # Create individual segment files
                segment_files = []
                for i, (start, end) in enumerate(segments):
                    segment_file = src.with_stem(f"segment_{i}_{uuid.uuid4().hex}")
                    print(f"DEBUG: Creating segment {i}: {start:.2f}s to {end:.2f}s")
                    
                    self.run([
                        "ffmpeg", "-y", "-i", str(src),
                        "-ss", f"{start}", "-to", f"{end}",
                        "-c:v", "libx264", "-c:a", "aac",
                        "-avoid_negative_ts", "make_zero",
                        str(segment_file)
                    ])
                    segment_files.append(segment_file)
                
                # Create concat list for segments
                concat_list = src.with_suffix(".txt")
                with concat_list.open("w") as f:
                    for segment_file in segment_files:
                        f.write(f"file '{segment_file.resolve()}'\n")
                
                # Concatenate segments and apply speed adjustment
                print("DEBUG: Concatenating segments and applying 1.25x speed...")
                self.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-filter_complex", "[0:v]setpts=0.8*PTS[v];[0:a]atempo=1.25[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(out)
                ])
                
                # Clean up segment files and concat list
                for segment_file in segment_files:
                    segment_file.unlink(missing_ok=True)
                concat_list.unlink(missing_ok=True)
            else:
                # No silence to remove, just apply speed adjustment
                print("DEBUG: No significant silence found, just applying speed adjustment...")
                self.run([
                    "ffmpeg", "-y", "-i", str(src),
                    "-filter_complex", "[0:v]setpts=0.8*PTS[v];[0:a]atempo=1.25[a]",
                    "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(out)
                ])
        else:
            # No silence detected, just apply speed adjustment
            print("DEBUG: No silence detected, just applying speed adjustment...")
            self.run([
                "ffmpeg", "-y", "-i", str(src),
                "-filter_complex", "[0:v]setpts=0.8*PTS[v];[0:a]atempo=1.25[a]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-c:a", "aac",
                str(out)
            ])
        
        # Verify output
        if out.exists():
            print(f"DEBUG: Processed video created: {out.stat().st_size} bytes")
        else:
            print("DEBUG: ERROR - Processed video was not created!")
        
        self._update_progress("Processing complete", 70)
        return out

    def add_flash_transition(self, hook, main):
        """Add flash transition between hook and main video with audio"""
        print("DEBUG: Adding flash transition")
        self._update_progress("Adding flash transition", 75)
        
        # Create a white flash clip (0.1 seconds)
        flash_clip = main.with_stem("flash_" + uuid.uuid4().hex)
        print(f"DEBUG: Creating flash clip: {flash_clip}")
        
        # Get video dimensions and frame rate from hook
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
            str(hook)
        ]
        result = self.run(probe_cmd)
        video_info = json.loads(result.stdout)
        
        # Find video stream
        video_stream = None
        for stream in video_info["streams"]:
            if stream["codec_type"] == "video":
                video_stream = stream
                break
        
        if video_stream:
            width = video_stream["width"]
            height = video_stream["height"]
            fps = video_stream.get("r_frame_rate", "30/1")
            print(f"DEBUG: Video dimensions: {width}x{height}, fps: {fps}")
        else:
            width, height, fps = 1920, 1080, "30/1"
            print("DEBUG: Using fallback dimensions")
        
        # Create white flash with silent audio track
        self.run([
            "ffmpeg", "-y", 
            "-f", "lavfi", "-i", f"color=white:size={width}x{height}:duration=0.1:rate={fps}",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t", "0.1",
            "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
            str(flash_clip)
        ])
        
        print(f"DEBUG: Flash clip created: {flash_clip.stat().st_size} bytes")
        return flash_clip

    def burn_captions(self, src, segments, speed_factor=1.0):
        """Burn captions into video with proper font and TikTok style - FIXED VERSION"""
        print(f"DEBUG: Burning captions for {len(segments)} segments with speed_factor={speed_factor}")
        self._update_progress("Generating and burning captions", 80)
        
        srt_file = src.with_suffix(".srt")
        print(f"DEBUG: Creating SRT file: {srt_file}")
        
        with srt_file.open("w", encoding="utf-8") as f:
            write_srt_tiktok_style(segments, f, speed_factor=speed_factor)
        
        # Verify SRT file was created
        if srt_file.exists():
            srt_content = srt_file.read_text()
            print(f"DEBUG: SRT file created with {len(srt_content)} characters")
            print(f"DEBUG: First 300 chars of SRT: {srt_content[:300]}...")
        else:
            print("DEBUG: ERROR - SRT file was not created!")
        
        font_file = Path("assets/ObelixProB-cyr.ttf").resolve()
        print(f"DEBUG: Checking font file: {font_file}")
        print(f"DEBUG: Font file exists: {font_file.exists()}")
        
        out = src.with_stem(src.stem + "_captioned")
        print(f"DEBUG: Output file will be: {out}")

        if font_file.exists():
            print("DEBUG: Using custom font with TikTok styling")
            print("DEBUG: About to start ffmpeg caption burning...")
            
            # SIMPLIFIED style - much less complex
            style = "Fontsize=48,PrimaryColour=&H00FFFFFF,Bold=1,Outline=2,Alignment=2,MarginV=50"
            
            # Try multiple encoder options with fallbacks
            encoders_to_try = [
                # Mac-friendly options first
                {"video": "libx264", "preset": ["-preset", "faster"]},  # Faster preset
                {"video": "libx264", "preset": ["-preset", "fast"]},    # Original fallback
            ]
            
            success = False
            last_error = None
            
            for i, encoder_config in enumerate(encoders_to_try):
                try:
                    print(f"DEBUG: Trying encoder option {i+1}: {encoder_config['video']}")
                    
                    cmd = [
                        "ffmpeg", "-y", 
                        "-i", str(src),
                        "-vf", f"subtitles={str(srt_file)}:fontsdir={str(font_file.parent)}:force_style='{style}'",
                        "-c:v", encoder_config["video"],
                        *encoder_config["preset"],
                        "-c:a", "copy",
                        str(out)
                    ]
                    
                    print(f"DEBUG: Command: {' '.join(cmd)}")
                    print("DEBUG: Starting caption burning...")
                    start_time = time.time()
                    
                    # Use longer timeout for caption burning as it's CPU intensive
                    self.run(cmd, timeout=600)  # 10 minute timeout
                    
                    duration = time.time() - start_time
                    print(f"DEBUG: Caption burning completed successfully in {duration:.2f}s")
                    success = True
                    break
                    
                except Exception as e:
                    print(f"DEBUG: Encoder {encoder_config['video']} failed: {str(e)}")
                    last_error = e
                    continue
            
            if not success:
                print("DEBUG: All encoders failed, trying minimal command...")
                # Last resort: minimal command without hardware acceleration
                try:
                    cmd = [
                        "ffmpeg", "-y", 
                        "-i", str(src),
                        "-vf", f"subtitles={str(srt_file)}",  # No custom font/style
                        "-c:v", "libx264",
                        "-c:a", "copy",
                        str(out)
                    ]
                    
                    print(f"DEBUG: Minimal command: {' '.join(cmd)}")
                    self.run(cmd, timeout=600)
                    print("DEBUG: Minimal caption burning succeeded")
                    success = True
                    
                except Exception as e:
                    print(f"DEBUG: Even minimal command failed: {str(e)}")
                    last_error = e
            
            if not success:
                raise RuntimeError(f"All caption burning methods failed. Last error: {last_error}")
                
        else:
            print("DEBUG: ERROR - Font file not found, using default font")
            # Fallback without custom font
            cmd = [
                "ffmpeg", "-y", 
                "-i", str(src),
                "-vf", f"subtitles={str(srt_file)}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "copy",
                str(out)
            ]
            
            self.run(cmd, timeout=600)
        
        # Verify output
        if out.exists():
            print(f"DEBUG: Captioned video created: {out.stat().st_size} bytes")
        else:
            print("DEBUG: ERROR - Captioned video was not created!")
        
        # Clean up SRT file
        srt_file.unlink(missing_ok=True)
        print("DEBUG: SRT file cleaned up")
        
        self._update_progress("Captions burned successfully", 85)
        return out

    def concat_clips_with_transition(self, hook, flash, main, out):
        """Concatenate hook, flash transition, and main video with format normalization"""
        print(f"DEBUG: Concatenating {hook}, {flash}, and {main} into {out}")
        self._update_progress("Combining hook, transition, and main video", 90)
        
        # Verify input files exist
        for file_path, name in [(hook, "hook"), (flash, "flash"), (main, "main")]:
            if not file_path.exists():
                print(f"DEBUG: ERROR - {name} file doesn't exist: {file_path}")
                raise FileNotFoundError(f"{name} file not found: {file_path}")
            print(f"DEBUG: {name} file size: {file_path.stat().st_size} bytes")
        
        # Get reference video properties from main video
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
            str(main)
        ]
        result = self.run(probe_cmd)
        video_info = json.loads(result.stdout)
        
        video_stream = None
        audio_stream = None
        for stream in video_info["streams"]:
            if stream["codec_type"] == "video" and video_stream is None:
                video_stream = stream
            elif stream["codec_type"] == "audio" and audio_stream is None:
                audio_stream = stream
        
        if video_stream:
            width = video_stream["width"]
            height = video_stream["height"]
            fps = video_stream.get("r_frame_rate", "30/1")
            print(f"DEBUG: Target format: {width}x{height}@{fps}")
        else:
            width, height, fps = 1920, 1080, "30/1"
            print("DEBUG: Using fallback format")
        
        # Use a more robust concatenation approach with format normalization
        print("DEBUG: Using robust concatenation with format normalization...")
        
        try:
            self.run([
                "ffmpeg", "-y", 
                "-i", str(hook), "-i", str(flash), "-i", str(main),
                "-filter_complex", 
                f"[0:v]scale={width}:{height},fps={fps},format=yuv420p[v0];"
                f"[1:v]scale={width}:{height},fps={fps},format=yuv420p[v1];"
                f"[2:v]scale={width}:{height},fps={fps},format=yuv420p[v2];"
                "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a0];"
                "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[a1];"
                "[2:a]aformat=sample_rates=48000:channel_layouts=stereo[a2];"
                "[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[outv][outa]",
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
                "-movflags", "+faststart",
                str(out)
            ], timeout=300)
            
        except Exception as e:
            print(f"DEBUG: Complex concatenation failed: {e}")
            print("DEBUG: Falling back to simple file-based concatenation...")
            
            # Fallback: normalize each file first, then concatenate
            normalized_files = []
            
            for i, (input_file, name) in enumerate([(hook, "hook"), (flash, "flash"), (main, "main")]):
                normalized_file = input_file.with_stem(f"normalized_{i}_{uuid.uuid4().hex}")
                print(f"DEBUG: Normalizing {name} to {normalized_file}")
                
                self.run([
                    "ffmpeg", "-y", "-i", str(input_file),
                    "-vf", f"scale={width}:{height},fps={fps},format=yuv420p",
                    "-af", "aformat=sample_rates=48000:channel_layouts=stereo",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(normalized_file)
                ])
                normalized_files.append(normalized_file)
            
            # Create concat file
            concat_list = main.with_suffix(".txt")
            with concat_list.open("w") as f:
                for norm_file in normalized_files:
                    f.write(f"file '{norm_file.resolve()}'\n")
            
            # Concatenate normalized files
            self.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                    "-c", "copy", "-movflags", "+faststart",
                    str(out)
            ])
                
            # Clean up normalized files and concat list
            for norm_file in normalized_files:
                norm_file.unlink(missing_ok=True)
            concat_list.unlink(missing_ok=True)
    
        # Verify final output
        if out.exists():
            print(f"DEBUG: Final video created: {out.stat().st_size} bytes")
        else:
            print("DEBUG: ERROR - Final video was not created!")
        
        # Clean up temp files
        hook.unlink(missing_ok=True)
        flash.unlink(missing_ok=True)
        main.unlink(missing_ok=True)
        print("DEBUG: Temp files cleaned up")
        
        self._update_progress("Video processing complete", 95)
    
    def _extract_clip_wrapper(self, src, start, end):
        """Thread-safe wrapper for extract_clip with timeout detection"""
        thread_id = threading.get_ident()
        start_time = time.time()
        print(f"DEBUG: [Thread-{thread_id}] Starting hook extraction at {start_time}")
        
        try:
            result = self.extract_clip(src, start, end)
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Hook extraction completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Hook extraction failed after {duration:.2f}s: {e}")
            raise
    
    def _remove_silence_wrapper(self, src):
        """Thread-safe wrapper for remove_silence with timeout detection"""
        thread_id = threading.get_ident()
        start_time = time.time()
        print(f"DEBUG: [Thread-{thread_id}] Starting silence removal at {start_time}")
        
        try:
            result = self.remove_silence(src)
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Silence removal completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            print(f"DEBUG: [Thread-{thread_id}] Silence removal failed after {duration:.2f}s: {e}")
            raise

    def process(self, source_file, output_file):
        """Main processing function with captions burned first to prevent desync"""
        try:
            src = Path(source_file).resolve()
            out = Path(output_file).resolve()
            
            print(f"DEBUG: [Main] Starting processing")
            print(f"DEBUG: [Main] Source: {src}")
            print(f"DEBUG: [Main] Output: {out}")
            
            # Verify source file exists
            if not src.exists():
                raise FileNotFoundError(f"Source file not found: {src}")
            print(f"DEBUG: [Main] Source file size: {src.stat().st_size} bytes")
            
            # Step 1: Transcription (sequential)
            print("DEBUG: [Main] === STEP 1: TRANSCRIPTION ===")
            segments = self.transcribe(src)
            
            # Step 2: Burn captions on original video (sequential)  
            print("DEBUG: [Main] === STEP 2: BURN CAPTIONS ON ORIGINAL VIDEO ===")
            captioned_src = self.burn_captions(src, segments, speed_factor=1.0)
            
            # Step 3: Hook detection (sequential)
            print("DEBUG: [Main] === STEP 3: HOOK DETECTION ===")
            hook_info = self.find_hook(segments)
            
            # Steps 4 & 5: Parallel processing with race condition detection
            print("DEBUG: [Main] === STEPS 4 & 5: PARALLEL PROCESSING ===")
            print("DEBUG: [Main] Creating thread pool executor...")
            
            hook_clip = None
            processed_main = None
            
            # Lower timeouts for faster failure detection
            hook_timeout = 120    # 2 minutes
            silence_timeout = 180 # 3 minutes
            
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    print("DEBUG: [Main] Submitting parallel tasks...")
                    
                    hook_future = executor.submit(
                        self._extract_clip_wrapper, 
                        captioned_src, hook_info["start"], hook_info["end"]  # Use captioned video
                    )
                    processed_future = executor.submit(
                        self._remove_silence_wrapper, 
                        captioned_src  # Use captioned video
                    )
                    
                    print("DEBUG: [Main] Waiting for hook extraction...")
                    start_time = time.time()
                    hook_clip = hook_future.result(timeout=hook_timeout)
                    hook_duration = time.time() - start_time
                    print(f"DEBUG: [Main] Hook extraction completed in {hook_duration:.2f}s")
                    
                    print("DEBUG: [Main] Waiting for silence removal...")
                    start_time = time.time()
                    processed_main = processed_future.result(timeout=silence_timeout)
                    silence_duration = time.time() - start_time
                    print(f"DEBUG: [Main] Silence removal completed in {silence_duration:.2f}s")
                    
            except concurrent.futures.TimeoutError as e:
                print("DEBUG: [Main] === TIMEOUT DETECTED ===")
                print(f"DEBUG: [Main] Hook timeout: {hook_timeout}s, Silence timeout: {silence_timeout}s")
                
                # Check which operation timed out
                if not hook_future.done():
                    print("DEBUG: [Main] Hook extraction timed out")
                if not processed_future.done():
                    print("DEBUG: [Main] Silence removal timed out")
                    
                raise RuntimeError(f"Video processing timed out: {str(e)}")
            
            # Verify parallel processing results with detailed error info
            if not hook_clip or not hook_clip.exists():
                print("DEBUG: [Main] === HOOK EXTRACTION VERIFICATION FAILED ===")
                print(f"DEBUG: [Main] hook_clip is None: {hook_clip is None}")
                if hook_clip:
                    print(f"DEBUG: [Main] hook_clip exists: {hook_clip.exists()}")
                    print(f"DEBUG: [Main] hook_clip path: {hook_clip}")
                raise RuntimeError("Hook extraction failed verification")
                
            if not processed_main or not processed_main.exists():
                print("DEBUG: [Main] === SILENCE REMOVAL VERIFICATION FAILED ===")
                print(f"DEBUG: [Main] processed_main is None: {processed_main is None}")
                if processed_main:
                    print(f"DEBUG: [Main] processed_main exists: {processed_main.exists()}")
                    print(f"DEBUG: [Main] processed_main path: {processed_main}")
                raise RuntimeError("Silence removal failed verification")
            
            # Step 6: Add flash transition
            print("DEBUG: [Main] === STEP 6: ADDING FLASH TRANSITION ===")
            flash_clip = self.add_flash_transition(hook_clip, processed_main)
            
            # Step 7: Final concatenation (no more caption burning needed)
            print("DEBUG: [Main] === STEP 7: FINAL CONCATENATION ===")
            self.concat_clips_with_transition(hook_clip, flash_clip, processed_main, out)
            
            # Clean up captioned source file
            captioned_src.unlink(missing_ok=True)
            print("DEBUG: [Main] Captioned source file cleaned up")
            
            final_metadata = {
                'processing_complete': True,
                'output_file': str(out),
                'total_segments': len(segments)
            }
            
            print("DEBUG: [Main] === PROCESSING COMPLETE ===")
            self._update_progress("Processing complete", 100, final_metadata)
            
        except Exception as e:
            print(f"DEBUG: [Main] === PROCESSING FAILED ===")
            print(f"DEBUG: [Main] Exception type: {type(e).__name__}")
            print(f"DEBUG: [Main] Exception message: {str(e)}")
            import traceback
            print(f"DEBUG: [Main] Traceback:\n{traceback.format_exc()}")
            
            # Check for potential race conditions in the error
            if "concurrent" in str(e).lower() or "lock" in str(e).lower():
                print("DEBUG: [Main] === POTENTIAL RACE CONDITION DETECTED ===")
                
            logging.error(f"Processing failed: {str(e)}")
            self._update_progress("Processing failed", 0, {'error': str(e)})
            raise
