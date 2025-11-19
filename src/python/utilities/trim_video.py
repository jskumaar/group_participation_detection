import os
import time
import subprocess
import sys # Used for clean exit

def trim_video(input_path: str, output_path: str, start_time_s: float, end_time_s: float):
    """
    Trims a video file, including the audio track, between specified start and end times 
    using a single, direct call to the FFmpeg executable.

    This method removes the dependency on the 'moviepy' library. 
    It maintains the original resolution and FPS by re-encoding, which is necessary 
    for perfect synchronization after time-based seeking.

    Args:
        input_path: Path to the input video file.
        output_path: Path to save the trimmed output video file.
        start_time_s: The time (in seconds) where the trimming should start (x s).
        end_time_s: The time (in seconds) where the trimming should end (y s).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Starting package-free video and audio trim for: {input_path}")
    print(f"Trim window: {start_time_s}s to {end_time_s}s")
    
    start_time = time.time()
    
    try:
        # Calculate the exact duration required for the -t flag
        duration_s = end_time_s - start_time_s
        
        if duration_s <= 0:
            print("Error: End time must be greater than start time.")
            return

        print("\n--- Executing FFmpeg Trim ---")
        
        # FFmpeg command structure:
        # 1. -ss {start}: Fast seek to the start time (applied before -i)
        # 2. -i {input}: Input file
        # 3. -t {duration}: Specifies the exact length of the output video
        # 4. -c:v libx264 / -c:a aac: Re-encode video and audio for synchronization
        #    (Necessary when using -ss before -i for accurate start points)
        
        cmd = [
            "ffmpeg",
            "-y",                          # Overwrite output file without asking
            
            # Seek flag BEFORE input for fast start point for both streams
            "-ss", str(start_time_s),      
            "-i", input_path,              
            
            "-t", str(duration_s),         # Cut the video after the duration
            
            # Video settings (maintaining resolution/FPS is automatic here)
            "-c:v", "libx264",             # Reliable H.264 video codec
            "-preset", "veryfast",         # Optimization for speed
            "-crf", "23",                  # Constant Rate Factor (quality setting)
            
            # Audio settings
            "-c:a", "aac",                 # Reliable AAC audio codec
            "-b:a", "192k",                # Audio bitrate
            
            # Map video and audio streams from the single input
            "-map", "0:v:0",
            "-map", "0:a:0",
            
            output_path
        ]

        print("Executing command:", " ".join(cmd))
        
        # Execute the FFmpeg command
        # check=True will raise an error if FFmpeg fails
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        total_time = time.time() - start_time
        
        print("\n" + "-" * 30)
        print(f"✅ Pure FFmpeg Trim successful!")
        print(f"Output saved to: {output_path}")
        print(f"Trimmed duration: {duration_s:.2f}s")
        print(f"Total processing time: {total_time:.2f} seconds.")
        
    except FileNotFoundError:
        print("❌ Critical Error: FFmpeg not found.")
        print("This script relies on the 'ffmpeg' program being installed and accessible in your system's PATH.")
        print("Please install FFmpeg to run this script.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Execution Error:")
        print(f"Command failed with exit code {e.returncode}.")
        print("FFmpeg Output (Stderr):")
        print(e.stderr)
        
    except Exception as e:
        print(f"❌ An unexpected Python error occurred: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Change 'input_video.mp4' to the path of your video file.
    INPUT_FILE = r"C:\Users\sures\Documents\Children_school_data\study4\children_school_study_4.mp4"
    OUTPUT_FILE = r"C:\Users\sures\Documents\Children_school_data\study4\children_school_study_4_interaction.mp4"

    # Define the trimming points: x s = 5.0, y s = 15.0
    # This will trim the video from the 5-second mark to the 15-second mark.
    START_TIME = 57.0
    END_TIME = 964.0
    # ---------------------

    if not os.path.exists(INPUT_FILE):
        print("="*70)
        print("NOTE: This version uses only FFmpeg (no MoviePy) to avoid dependency issues.")
        print("Please ensure the 'ffmpeg' executable is installed on your system.")
        print(f"ATTENTION: Input file '{INPUT_FILE}' not found.")
        print("Please replace 'input_video.mp4' with the actual path to your video.")
        print("="*70)
    else:
        trim_video(INPUT_FILE, OUTPUT_FILE, START_TIME, END_TIME)