import cv2
from pathlib import Path
import os

def create_video_from_screenshots(screenshots_dir, output_video, fps=5):
    """
    Create MP4 video from screenshots.
    
    Args:
        screenshots_dir: Path to folder containing iter_*.png files
        output_video: Output MP4 file path
        fps: Frames per second
    """
    screenshots_dir = Path(screenshots_dir)
    
    # Get all image files sorted by iteration number
    image_files = sorted(screenshots_dir.glob("iter_*.png"))
    
    if not image_files:
        print(f"No screenshots found in {screenshots_dir}")
        return
    
    print(f"Found {len(image_files)} screenshots")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    height, width = first_img.shape[:2]
    print(f"Video dimensions: {width}x{height}, fps: {fps}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    # Write all frames
    for i, img_path in enumerate(image_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read {img_path}")
            continue
        out.write(frame)
        print(f"Added frame {i+1}/{len(image_files)}")
    
    out.release()
    print(f"\nVideo saved to {output_video}")

if __name__ == "__main__":
    base_path = Path("/Users/aa13586/Desktop/symmetric-dirichlet")
    screenshots_dir = base_path / "screenshots"
    output_video = base_path / "optimization_animation.mp4"
    
    create_video_from_screenshots(screenshots_dir, output_video, fps=5)