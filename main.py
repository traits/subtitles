import json
import os
import subprocess
from pathlib import Path

import cv2


def extract_subtitles(mkv_file):
    # Create the output directory if it doesn't exist
    output_dir = Path("subpics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract video frames using ffmpeg
    subprocess.run(["ffmpeg", "-i", mkv_file, "-vf", "fps=1", str(output_dir / "frame%05d.png")], check=True)

    # Process each frame to detect and extract subtitles
    subtitles = []
    frame_count = 1
    for frame_path in sorted(output_dir.glob("frame*.png")):
        img = cv2.imread(str(frame_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Assuming subtitles are at the bottom center of the frame
        height, width = gray.shape
        subtitle_region = gray[int(height * 0.8) : height, int(width * 0.3) : int(width * 0.7)]

        # Threshold to isolate the subtitle text
        _, thresh = cv2.threshold(subtitle_region, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the bounding box for the largest contour (assuming it's the subtitle)
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

            # Adjust bounding box to original image coordinates
            x += int(width * 0.3)
            y += int(height * 0.8)

            # Crop the subtitle region
            subtitle_img = img[y : y + h, x : x + w]

            # Save the cropped subtitle image
            subtitle_path = output_dir / f"{frame_count:05d}.png"
            cv2.imwrite(str(subtitle_path), subtitle_img)

            # Store the timestamp and associated image path
            subtitles.append({"timestamp": f"frame{frame_count:05d}", "image": subtitle_path.name})

            frame_count += 1

    # Write the timestamps and associated image paths to a JSON file
    with open("subtitles.json", "w") as json_file:
        json.dump(subtitles, json_file, indent=4)


if __name__ == "__main__":
    mkv_file = "input.mkv"  # Replace with your MKV file path
    extract_subtitles(mkv_file)
