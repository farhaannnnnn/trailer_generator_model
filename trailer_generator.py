import cv2
import numpy as np
import os
import sys

# MoviePy import with friendly error message if missing
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except Exception as e:
    print("ERROR: MoviePy is required but couldn't be imported.")
    print("Install it with: python -m pip install moviepy imageio-ffmpeg")
    print("If you already installed it, ensure you're running the same Python interpreter where it was installed.")
    print(f"Import error detail: {type(e).__name__}: {e}")
    sys.exit(1)
# The pydub imports have been removed to fix the ModuleNotFoundError
# from pydub import AudioSegment 
# from pydub.utils import mediainfo 

# --- CONFIGURATION ---
INPUT_VIDEO_PATH = "your_movie.mp4" # <--- REPLACE WITH YOUR MOVIE FILE NAME
OUTPUT_TRAILER_PATH = "generated_trailer.mp4"
CLIP_DURATION_SECONDS = 2.0      # Duration of each segment for analysis
TRAILER_LENGTH_SECONDS = 30.0    # Desired final trailer length

# --- STEP 1: PREPARATION & SEGMENTATION ---
def get_segments(video_path, duration):
    """Calculates the start/end times for all fixed-length segments."""
    try:
        clip = VideoFileClip(video_path)
        total_duration = clip.duration
        clip.close()
    except Exception as e:
        print(f"Error loading video file with moviepy: {e}")
        # Ensure the clip is closed even if an error occurs during processing
        try:
            clip.close() 
        except:
            pass
        return []

    segments = []
    start_time = 0.0
    while start_time + duration <= total_duration:
        segments.append({'start': start_time, 'end': start_time + duration, 'score': 0.0})
        start_time += duration
    print(f"Total movie duration: {total_duration:.2f}s. Generated {len(segments)} segments.")
    return segments

# --- STEP 2: FEATURE SCORING (SIMPLIFIED DEEP LEARNING PROXY) ---

def calculate_visual_score(video_path, start_time, end_time):
    """
    Calculates the 'motion' score by averaging the absolute difference between successive frames.
    (Proxy for CNN/Spatiotemporal Feature Importance)
    """
    cap = cv2.VideoCapture(video_path)
    # Set the starting position in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    motion_scores = []
    prev_frame_gray = None
    
    # Process frames until the end time
    while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time * 1000:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for faster processing
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame_gray is not None:
            # Calculate absolute difference (Motion proxy)
            diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
            motion_score = np.mean(diff)
            motion_scores.append(motion_score)
        
        prev_frame_gray = current_frame_gray
    
    cap.release()
    if not motion_scores:
        return 0.0
        
    # The higher the average difference, the more motion/change occurred
    return np.mean(motion_scores)

def calculate_audio_score(video_path, start_time, end_time):
    """
    Calculates the 'impact' score based on audio energy (loudness) in the segment.
    Uses moviepy to extract audio and numpy to calculate RMS.
    """
    
    try:
        # Load the full audio from the video file
        with VideoFileClip(video_path) as clip:
            if clip.audio is None:
                # If there is no audio track, return 0 score
                return 0.0

            # Extract the subclip's audio data as a numpy array
            audio_subclip = clip.audio.subclip(start_time, end_time)

            # MoviePy provides `to_soundarray()` which returns a numpy array of shape
            # (n_samples, n_channels). The values are usually floats in [-1, 1],
            # but some backends may return integer arrays (e.g. int16).
            try:
                audio_array = audio_subclip.to_soundarray()
            except AttributeError:
                # Older/newer MoviePy versions may expose a slightly different name
                # or the method may require an fps argument. Try a couple fallbacks.
                try:
                    audio_array = audio_subclip.to_soundarray(fps=44100)
                except Exception:
                    # As a last resort, convert via numpy from the raw audio
                    audio_array = np.array(audio_subclip.to_soundarray())

            audio_array = np.asarray(audio_array)
            if audio_array.size == 0:
                return 0.0

            # RMS calculation (work in float64 for stability)
            rms_score = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))

            # Determine sensible normalization:
            # - If the audio data are float in [-1, 1], the peak is 1.0
            # - If the audio data are int16, the peak will be around 32767
            peak = np.max(np.abs(audio_array))
            if peak <= 0:
                normalized_score = 0.0
            else:
                normalized_score = float(rms_score / peak)

            # Clamp to [0, 1]
            normalized_score = float(np.clip(normalized_score, 0.0, 1.0))

            return normalized_score

    except Exception as e:
        # Provide more context in the warning to help debugging (exception type + message)
        print(f"Warning: Audio scoring failed for segment at {start_time}s. Using score 0.0. Error: {type(e).__name__}: {e}")
        return 0.0

def score_all_segments(video_path, segments):
    """Iterates through all segments and assigns a combined importance score."""
    for i, segment in enumerate(segments):
        print(f"Scoring segment {i+1}/{len(segments)} ({segment['start']:.2f}s - {segment['end']:.2f}s)...")
        
        # Score 1: Visual Motion (High motion = High score)
        v_score = calculate_visual_score(video_path, segment['start'], segment['end'])
        
        # Score 2: Audio Impact (High loudness = High score)
        a_score = calculate_audio_score(video_path, segment['start'], segment['end'])
        
        # Combine the scores (Simple Weighted Average)
        # Weighting motion slightly higher to focus on visual change
        combined_score = (v_score * 0.6) + (a_score * 0.4) 
        segment['score'] = combined_score
        
    # Normalize all scores globally (0 to 1) for better comparison
    all_scores = [s['score'] for s in segments]
    max_score = max(all_scores) if all_scores else 1.0
    # Prevent division by zero if all scores are 0 (e.g., in a silent, static video)
    if max_score > 0:
        for segment in segments:
            segment['score'] /= max_score
            
    return segments

# --- STEP 3: KEY MOMENT SELECTION ---

def select_key_moments(segments, trailer_length, clip_duration):
    """Sorts and selects the top-scoring, non-redundant segments."""
    
    # 1. Calculate number of clips needed
    num_clips = int(trailer_length / clip_duration)
    print(f"\nSelecting top {num_clips} clips for a {trailer_length}s trailer.")

    # 2. Sort by score (Highest score first)
    sorted_segments = sorted(segments, key=lambda x: x['score'], reverse=True)
    
    # 3. Select the top N clips
    selected_clips = sorted_segments[:num_clips]
    
    # 4. Re-sort the selected clips by their original timeline position 
    # to maintain narrative coherence.
    selected_clips.sort(key=lambda x: x['start'])
    
    top_scores_list = ["{:.4f}".format(c['score']) for c in selected_clips]
    print(f"Selection complete. Top scores: {top_scores_list}")
    
    return selected_clips

# --- STEP 4: TRAILER ASSEMBLY ---

def assemble_trailer(video_path, selected_clips, output_path):
    """Stitches the selected clips into the final video file using MoviePy."""
    if not selected_clips:
        print("No clips selected. Assembly aborted.")
        return

    print("\n--- Starting Trailer Assembly ---")
    
    try:
        # Load the full video once
        full_clip = VideoFileClip(video_path)
        
        # Create sub-clips based on selected segments' start/end times
        trailer_clips = []
        for i, segment in enumerate(selected_clips):
            # subclip must be done on the video stream itself
            subclip = full_clip.subclip(segment['start'], segment['end'])
            trailer_clips.append(subclip)
            print(f"Assembling clip {i+1}/{len(selected_clips)} from {segment['start']:.2f}s to {segment['end']:.2f}s...")
        
        # Concatenate all sub-clips
        final_trailer = concatenate_videoclips(trailer_clips)
        
        # Write the final file
        final_trailer.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac", 
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True,
            fps=full_clip.fps,
            preset="fast" # Use 'medium' or 'slow' for higher quality final production
        )

        full_clip.close()
        print(f"\n--- SUCCESS: Trailer saved to {OUTPUT_TRAILER_PATH} ---")

    except Exception as e:
        print(f"Assembly Error: {e}")
        print("Ensure the video file is not corrupted and FFMPEG is installed and in your system PATH.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"ERROR: Input video not found at '{INPUT_VIDEO_PATH}'. Please update the path.")
    else:
        # 1. Segment the movie
        segments = get_segments(INPUT_VIDEO_PATH, CLIP_DURATION_SECONDS)
        
        # 2. Score all segments
        scored_segments = score_all_segments(INPUT_VIDEO_PATH, segments)
        
        # 3. Select key moments
        key_moments = select_key_moments(scored_segments, TRAILER_LENGTH_SECONDS, CLIP_DURATION_SECONDS)
        
        # 4. Assemble and output the trailer
        assemble_trailer(INPUT_VIDEO_PATH, key_moments, OUTPUT_TRAILER_PATH)
