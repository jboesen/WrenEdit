import subprocess
from pathlib import Path
import uuid
import logging
import time
import json
import whisper
from transformers import pipeline

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

def write_srt_tiktok_style(segments, file, speed_factor=1.25):
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
        self._load_models()
        
    def _update_progress(self, step, progress, metadata=None):
        """Update progress with callback"""
        if self.progress_callback:
            self.progress_callback(step, progress, metadata)
        logging.info(f"Progress: {step} - {progress}%")
        print(f"DEBUG: {step} - {progress}%")
        if metadata:
            print(f"DEBUG: Metadata: {metadata}")
    
    def _load_models(self):
        """Load AI models"""
        print("DEBUG: Starting model loading...")
        self._update_progress("Loading emotion analysis model", 5)
        self.emotion = pipeline(
            "text-classification",
            model="tasinhoque/text-classification-goemotions",
            top_k=None,
            truncation=True,
        )
        print("DEBUG: Emotion model loaded")
        
        self._update_progress("Loading speech recognition model", 15)
        self.model = whisper.load_model("medium")
        print("DEBUG: Whisper model loaded")
    
    def run(self, cmd):
        """Run subprocess command with extensive debugging"""
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"DEBUG: Command succeeded")
            if result.stdout:
                print(f"DEBUG: STDOUT: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            print(f"DEBUG: Command FAILED: {' '.join(cmd)}")
            print(f"DEBUG: Return code: {e.returncode}")
            print(f"DEBUG: STDERR: {e.stderr}")
            print(f"DEBUG: STDOUT: {e.stdout}")
            logging.error(f"Command failed: {' '.join(cmd)}")
            logging.error(f"Error: {e.stderr}")
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
        """Find the best hook segment based on emotion analysis"""
        print(f"DEBUG: Finding hook in {len(segments)} segments (duration {min_s}-{max_s}s)")
        self._update_progress("Analyzing emotions for hook detection", 40)
        
        best = None
        analyzed_segments = []
        
        for i, seg in enumerate(segments):
            dur = seg["end"] - seg["start"]
            if not (min_s <= dur <= max_s):
                continue
            
            print(f"DEBUG: Analyzing segment {i}: {dur:.2f}s - '{seg['text'][:30]}...'")
            scores = self.emotion(seg["text"])[0]
            score = sum(
                item["score"]
                for item in scores
                if item["label"] in {"joy", "amusement", "surprise"}
            )
            
            print(f"DEBUG: Emotion score: {score:.3f}")
            
            analyzed_segments.append({
                'text': seg["text"],
                'duration': dur,
                'emotion_score': score,
                'start': seg["start"],
                'end': seg["end"]
            })
            
            if best is None or score > best["score"]:
                best = {"start": seg["start"], "end": seg["end"], "score": score, "text": seg["text"]}
                print(f"DEBUG: New best hook found with score {score:.3f}")
        
        if best is None:
            print("DEBUG: No suitable hook found, using fallback")
            # fallback: first sentence under max_s
            for seg in segments:
                if seg["end"] - seg["start"] <= max_s:
                    best = {"start": seg["start"], "end": seg["end"], "score": 0.0, "text": seg["text"]}
                    print(f"DEBUG: Using fallback hook: {best['start']:.2f}-{best['end']:.2f}")
                    break
        
        if best:
            print(f"DEBUG: Final hook selected: {best['start']:.2f}-{best['end']:.2f} (score: {best['score']:.3f})")
            print(f"DEBUG: Hook text: {best['text']}")
        
        hook_metadata = {
            'hook_segment': {
                'start': best["start"] if best else 0,
                'end': best["end"] if best else 0,
                'text': best.get("text", "") if best else "",
                'emotion_score': best["score"] if best else 0
            },
            'analyzed_segments': analyzed_segments[:5]  # Top 5 for preview
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
        """Remove silence from video and speed up to 1.25x"""
        print(f"DEBUG: Processing video for silence removal and speed adjustment")
        self._update_progress("Removing silence and adjusting speed", 65)
        
        out = src.with_stem(src.stem + "_processed")
        print(f"DEBUG: Output file: {out}")
        
        # Separate audio and video processing to avoid sync issues
        # First process audio
        audio_temp = src.with_stem(src.stem + "_audio_temp").with_suffix(".wav")
        print("DEBUG: Processing audio...")
        self.run([
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-ar", "44100", "-ac", "2",
            str(audio_temp)
        ])
        
        # Process audio: remove silence and adjust tempo
        audio_processed = src.with_stem(src.stem + "_audio_processed").with_suffix(".wav")
        print("DEBUG: Removing silence and adjusting tempo...")
        self.run([
            "ffmpeg", "-y", "-i", str(audio_temp),
            "-af", "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-35dB,atempo=1.25",
            str(audio_processed)
        ])
        
        # Speed up video to match
        print("DEBUG: Speeding up video...")
        self.run([
            "ffmpeg", "-y", "-i", str(src), "-i", str(audio_processed),
            "-filter_complex", "[0:v]setpts=0.8*PTS[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            str(out)
        ])
        
        # Clean up temp files
        audio_temp.unlink(missing_ok=True)
        audio_processed.unlink(missing_ok=True)
        
        # Verify output
        if out.exists():
            print(f"DEBUG: Processed video created: {out.stat().st_size} bytes")
        else:
            print("DEBUG: ERROR - Processed video was not created!")
        
        self._update_progress("Processing complete", 70)
        return out

    def add_flash_transition(self, hook, main):
        """Add flash transition between hook and main video"""
        print("DEBUG: Adding flash transition")
        self._update_progress("Adding flash transition", 75)
        
        # Create a white flash clip (0.1 seconds)
        flash_clip = main.with_stem("flash_" + uuid.uuid4().hex)
        print(f"DEBUG: Creating flash clip: {flash_clip}")
        
        # Get video dimensions from hook
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
            print(f"DEBUG: Video dimensions: {width}x{height}")
        else:
            width, height = 1920, 1080  # fallback
            print("DEBUG: Using fallback dimensions")
        
        # Create white flash
        self.run([
            "ffmpeg", "-y", "-f", "lavfi", 
            "-i", f"color=white:size={width}x{height}:duration=0.1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(flash_clip)
        ])
        
        print(f"DEBUG: Flash clip created: {flash_clip.stat().st_size} bytes")
        return flash_clip

    def burn_captions(self, src, segments):
        """Burn captions into video with proper font and TikTok style"""
        print(f"DEBUG: Burning captions for {len(segments)} segments")
        self._update_progress("Generating and burning captions", 80)
        
        srt_file = src.with_suffix(".srt")
        print(f"DEBUG: Creating SRT file: {srt_file}")
        
        with srt_file.open("w", encoding="utf-8") as f:
            write_srt_tiktok_style(segments, f, speed_factor=1.25)
        
        # Verify SRT file was created
        if srt_file.exists():
            srt_content = srt_file.read_text()
            print(f"DEBUG: SRT file created with {len(srt_content)} characters")
            print(f"DEBUG: First 300 chars of SRT: {srt_content[:300]}...")
        else:
            print("DEBUG: ERROR - SRT file was not created!")
        
        font_file = Path("assets/fonts/ObelixProB-cyr.ttf").resolve()
        print(f"DEBUG: Checking font file: {font_file}")
        print(f"DEBUG: Font file exists: {font_file.exists()}")
        
        out = src.with_stem(src.stem + "_captioned")
        print(f"DEBUG: Output file will be: {out}")

        if font_file.exists():
            print("DEBUG: Using custom font with TikTok styling")
            style = (
                f"FontName={font_file.stem},"
                "Fontsize=48,"                      # Bigger font
                "PrimaryColour=&H00FFFFFF,"         # White text
                "SecondaryColour=&H00000000,"       # Black outline
                "OutlineColour=&H00000000,"         # Black outline
                "BackColour=&H80000000,"            # Semi-transparent background
                "Bold=1,"                           # Bold text
                "Italic=0,"
                "Underline=0,"
                "StrikeOut=0,"
                "ScaleX=100,"
                "ScaleY=100,"
                "Spacing=0,"
                "Angle=0,"
                "BorderStyle=3,"                    # Box background
                "Outline=2,"                        # Thick outline
                "Shadow=0,"
                "Alignment=2,"                      # Bottom center
                "MarginL=0,"
                "MarginR=0,"
                "MarginV=50"                        # Bottom margin
            )
            
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-vf", f"subtitles={str(srt_file)}:fontsdir={font_file.parent}:force_style='{style}'",
                "-c:a", "copy",
                str(out)
            ]
        else:
            print("DEBUG: ERROR - Font file not found, this should not happen!")
            print(f"DEBUG: Tried to find font at: {font_file}")
            # Don't use fallback, let it fail so we can debug
            raise FileNotFoundError(f"Font file not found: {font_file}")
        
        self.run(cmd)
        
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
        """Concatenate hook, flash transition, and main video"""
        print(f"DEBUG: Concatenating {hook}, {flash}, and {main} into {out}")
        self._update_progress("Combining hook, transition, and main video", 90)
        
        # Verify input files exist
        for file_path, name in [(hook, "hook"), (flash, "flash"), (main, "main")]:
            if not file_path.exists():
                print(f"DEBUG: ERROR - {name} file doesn't exist: {file_path}")
                raise FileNotFoundError(f"{name} file not found: {file_path}")
            print(f"DEBUG: {name} file size: {file_path.stat().st_size} bytes")
        
        lst = main.with_suffix(".txt")
        concat_list = f"file '{hook.resolve()}'\nfile '{flash.resolve()}'\nfile '{main.resolve()}'\n"
        lst.write_text(concat_list)
        print(f"DEBUG: Concat list created: {concat_list}")
        
        self.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c", "copy", str(out)
        ])
        
        # Verify final output
        if out.exists():
            print(f"DEBUG: Final video created: {out.stat().st_size} bytes")
        else:
            print("DEBUG: ERROR - Final video was not created!")
        
        # Clean up temp files
        lst.unlink(missing_ok=True)
        hook.unlink(missing_ok=True)
        flash.unlink(missing_ok=True)
        main.unlink(missing_ok=True)
        print("DEBUG: Temp files cleaned up")
        
        self._update_progress("Video processing complete", 95)
    
    def process(self, source_file, output_file):
        """Main processing function"""
        try:
            src = Path(source_file).resolve()
            out = Path(output_file).resolve()
            
            print(f"DEBUG: Starting processing")
            print(f"DEBUG: Source: {src}")
            print(f"DEBUG: Output: {out}")
            
            # Verify source file exists
            if not src.exists():
                raise FileNotFoundError(f"Source file not found: {src}")
            print(f"DEBUG: Source file size: {src.stat().st_size} bytes")
            
            # Transcribe video
            print("DEBUG: === STEP 1: TRANSCRIPTION ===")
            segments = self.transcribe(src)
            
            # Find hook
            print("DEBUG: === STEP 2: HOOK DETECTION ===")
            hook_info = self.find_hook(segments)
            if not hook_info:
                raise ValueError("Could not find suitable hook segment")
            
            # Extract hook clip
            print("DEBUG: === STEP 3: HOOK EXTRACTION ===")
            hook_clip = self.extract_clip(src, hook_info["start"], hook_info["end"])
            
            # Remove silence from main video and speed up
            print("DEBUG: === STEP 4: SILENCE REMOVAL & SPEED ADJUSTMENT ===")
            processed_main = self.remove_silence(src)
            
            # Add flash transition
            print("DEBUG: === STEP 5: ADDING FLASH TRANSITION ===")
            flash_clip = self.add_flash_transition(hook_clip, processed_main)
            
            # Burn captions (LAST STEP to ensure proper timing)
            print("DEBUG: === STEP 6: CAPTION BURNING ===")
            captioned_main = self.burn_captions(processed_main, segments)
            
            # Concatenate with transition
            print("DEBUG: === STEP 7: CONCATENATION WITH TRANSITION ===")
            self.concat_clips_with_transition(hook_clip, flash_clip, captioned_main, out)
            
            # Final metadata
            final_metadata = {
                'processing_complete': True,
                'output_file': str(out),
                'total_segments': len(segments)
            }
            
            print("DEBUG: === PROCESSING COMPLETE ===")
            self._update_progress("Processing complete", 100, final_metadata)
            
        except Exception as e:
            print(f"DEBUG: === PROCESSING FAILED ===")
            print(f"DEBUG: Exception type: {type(e).__name__}")
            print(f"DEBUG: Exception message: {str(e)}")
            import traceback
            print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
            logging.error(f"Processing failed: {str(e)}")
            self._update_progress("Processing failed", 0, {'error': str(e)})
            raise
