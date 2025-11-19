#!/usr/bin/env python3
"""
Video Framerate Adjuster
Adjusts the framerate of a video to match a target duration.
"""

import cv2
import sys
import argparse
from pathlib import Path


def get_video_info(video_path):
    """Get video information including fps, frame count, and duration."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }


def adjust_framerate(input_path, output_path, target_duration):
    """
    Adjust video framerate to match target duration.
    
    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        target_duration: Desired duration in seconds
    """
    # Get input video information
    video_info = get_video_info(input_path)
    
    print(f"\n{'='*50}")
    print(f"Input Video Information:")
    print(f"{'='*50}")
    print(f"Original FPS: {video_info['fps']:.3f}")
    print(f"Frame Count: {video_info['frame_count']}")
    print(f"Resolution: {video_info['width']}x{video_info['height']}")
    print(f"Current Duration: {video_info['duration']:.3f} seconds")
    print(f"Target Duration: {target_duration:.3f} seconds")
    
    # Calculate new framerate
    new_fps = video_info['frame_count'] / target_duration
    
    print(f"\n{'='*50}")
    print(f"Calculated new FPS: {new_fps:.3f}")
    print(f"{'='*50}\n")
    
    if new_fps <= 0:
        raise ValueError("Calculated FPS is invalid. Check target duration.")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change codec if needed
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        new_fps,
        (video_info['width'], video_info['height'])
    )
    
    # Read and write all frames
    frame_num = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        out.write(frame)
        frame_num += 1
        
        # Progress indicator
        if frame_num % 30 == 0:
            progress = (frame_num / video_info['frame_count']) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{video_info['frame_count']} frames)", end='\r')
    
    print(f"\nProgress: 100.0% ({frame_num}/{video_info['frame_count']} frames)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\n{'='*50}")
    print(f"✓ Video saved to: {output_path}")
    print(f"✓ New framerate: {new_fps:.3f} fps")
    print(f"✓ Duration: {target_duration:.3f} seconds")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Adjust video framerate to match a target duration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Change a 10-second video to 15 seconds
  python adjust_video_framerate.py input.mp4 output.mp4 15

  # Change a 30-second video to 1 minute (60 seconds)
  python adjust_video_framerate.py input.mp4 output.mp4 60
  
  # Show info about a video without processing
  python adjust_video_framerate.py input.mp4 --info
        """
    )
    
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', nargs='?', help='Output video file path')
    parser.add_argument('duration', nargs='?', type=float, 
                       help='Target duration in seconds')
    parser.add_argument('--info', action='store_true',
                       help='Show video information only')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    # Info mode
    if args.info:
        info = get_video_info(args.input)
        print(f"\n{'='*50}")
        print(f"Video Information: {args.input}")
        print(f"{'='*50}")
        print(f"FPS: {info['fps']:.3f}")
        print(f"Frame Count: {info['frame_count']}")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"Duration: {info['duration']:.3f} seconds ({info['duration']/60:.3f} minutes)")
        print(f"{'='*50}\n")
        return
    
    # Normal mode - require output and duration
    if not args.output or args.duration is None:
        parser.print_help()
        print("\nError: Both output file and duration are required!")
        sys.exit(1)
    
    try:
        adjust_framerate(args.input, args.output, args.duration)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()