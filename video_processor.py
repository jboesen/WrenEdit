import subprocess
from pathlib import Path
import uuid
import whisper
from transformers import pipeline
import logging
import time

class VideoProcessor:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.emotion = None
        self.model = None
        
    def _update_progress(self, step, progress, metadata=None):
        """Update progress with callback"""
        if self.progress_callback:
            self.progress_callback(step, progress, metadata)
        logging.info(f"Progress: {step} - {progress}%")
    
    def _load_models(self):
        """Load AI models if not already loaded"""
        if self.emotion is None:
            self._update_progress("Loading emotion analysis model", 5)
            self.emotion = pipeline(
                "text-classification",
                model="tasinhoque/text-classification-goemotions",
                top_k=None,
                truncation=True,
            )
        
        if self.model is None:
            self._update_progress("Loading speech recognition model", 15)
            self.model = whisper.load_model("medium")
    
    def run(self, cmd):
        """Run subprocess command"""
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {' '.join(cmd)}")
            logging.error(f"Error: {e.stderr}")
            raise
    
    def transcribe(self, video_path):
        """Transcribe video to get segments"""
        self._update_progress("Transcribing audio", 25)
        result = self.model.transcribe(str(video_path))
        segments = result["segments"]
        
        self._update_progress("Transcription complete", 35, {
            'segments_count': len(segments),
            'total_duration': max(seg['end'] for seg in segments) if segments else 0
        })
        
        return segments
    
    def find_hook(self, segments, min_s=3.0, max_s=6.0):
        """Find the best hook segment based on emotion analysis"""
        self._update_progress("Analyzing emotions for hook detection", 40)
        
        best = None
        analyzed_segments = []
        
        for i, seg in enumerate(segments):
            dur = seg["end"] - seg["start"]
            if not (min_s <= dur <= max_s):
                continue
            
            scores = self.emotion(seg["text"])[0]
            score = sum(
                item["score"]
                for item in scores
                if item["label"] in {"joy", "amusement", "surprise"}
            )
            
            analyzed_segments.append({
                'text': seg["text"],
                'duration': dur,
                'emotion_score': score,
                'start': seg["start"],
                'end': seg["end"]
            })
            
            if best is None or score > best["score"]:
                best = {"start": seg["start"], "end": seg["end"], "score": score, "text": seg["text"]}
        
        if best is None:
            # fallback: first sentence under max_s
            for seg in segments:
                if seg["end"] - seg["start"] <= max_s:
                    best = {"start": seg["start"], "end": seg["end"], "score": 0.0, "text": seg["text"]}
                    break
        
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
        self._update_progress("Extracting hook clip", 55)
        
        out = src.with_stem("hook_" + uuid.uuid4().hex)
        self.run([
            "ffmpeg", "-y", "-i", str(src), "-ss", f"{start}", "-to", f"{end}",
            "-c", "copy", str(out)
        ])
        
        self._update_progress("Hook clip extracted", 60)
        return out
    
    def remove_silence(self, src):
        """Remove silence from video"""
        self._update_progress("Removing silence", 65)
        
        out = src.with_stem(src.stem + "_nosilence")
        self.run([
            "ffmpeg", "-y", "-i", str(src), "-af",
            "silenceremove=start_periods=1:start_duration=0.3:start_threshold=-35dB",
            str(out)
        ])
        
        self._update_progress("Silence removal complete", 75)
        return out
    
    def burn_captions(self, src, segments):
        """Burn captions into video"""
        self._update_progress("Generating and burning captions", 80)
        
        srt = src.with_suffix(".srt")
        whisper.utils.write_srt(segments, file=srt.open("w", encoding="utf-8"))
        
        out = src.with_stem(src.stem + "_captioned")
        self.run([
            "ffmpeg", "-y", "-i", str(src), "-vf",
            f"subtitles={str(srt)}:force_style='Fontsize=48'",
            str(out)
        ])
        
        # Clean up SRT file
        srt.unlink(missing_ok=True)
        
        self._update_progress("Captions burned successfully", 90)
        return out
    
    def concat_clips(self, hook, main, out):
        """Concatenate hook and main video"""
        self._update_progress("Combining hook and main video", 95)
        
        lst = main.with_suffix(".txt")
        lst.write_text(f"file '{hook.resolve()}'\nfile '{main.resolve()}'\n")
        
        self.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c", "copy", str(out)
        ])
        
        # Clean up temp files
        lst.unlink(missing_ok=True)
        hook.unlink(missing_ok=True)
        main.unlink(missing_ok=True)
        
        self._update_progress("Video processing complete", 100)
    
    def process(self, source_file, output_file):
        """Main processing function"""
        try:
            src = Path(source_file).resolve()
            out = Path(output_file).resolve()
            
            # Load models
            self._load_models()
            
            # Transcribe video
            segments = self.transcribe(src)
            
            # Find hook
            hook_info = self.find_hook(segments)
            if not hook_info:
                raise ValueError("Could not find suitable hook segment")
            
            # Extract hook clip
            hook_clip = self.extract_clip(src, hook_info["start"], hook_info["end"])
            
            # Remove silence from main video
            trimmed = self.remove_silence(src)
            
            # Burn captions
            captioned = self.burn_captions(trimmed, segments)
            
            # Concatenate clips
            self.concat_clips(hook_clip, captioned, out)
            
            # Final metadata
            final_metadata = {
                'processing_complete': True,
                'output_file': str(out),
                'total_segments': len(segments)
            }
            
            self._update_progress("Processing complete", 100, final_metadata)
            
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            self._update_progress("Processing failed", 0, {'error': str(e)})
            raise
