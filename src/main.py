import os
import cv2
import fire
import subprocess
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from ultralytics import YOLO

def collect_video_paths(root_dir: str) -> List[Path]:
    """
    Recursively collect all .mp4 video file paths from a root directory.

    Args:
        root_dir (str): Root directory to search for video files.

    Returns:
        List[Path]: List of paths to video files.
    """
    root_path = Path(root_dir)
    return list(root_path.rglob("*.mp4"))

def process_videos(input_folder: str, output_folder: str, model_path: str) -> None:
    """
    Process all video files in the input folder using YOLO object detection.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to the folder where output images and logs will be saved.
        model_path (str): Path to the YOLO model (e.g., yolov8s.pt).
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file_path = output_path / "log.txt"

    yolo = YOLO(model_path)
    video_files = collect_video_paths(str(input_path))

    with open(log_file_path, "w") as log_file:
        for video_file in video_files:
            cap = cv2.VideoCapture(str(video_file))
            video_name = video_file.stem
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = yolo(frame)
                if not results:
                    continue
                result = results[0]

                # Draw results using YOLO's inbuilt visualizer
                annotated_frame = result.plot()
                frame_log: List[str] = []

                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        cls = int(box.cls[0])
                        label = result.names[cls] if cls in result.names else str(cls)
                        frame_log.append(f"Detected {label} with confidence {box.conf[0]:.2f}")

                # Save frame image
                frame_filename = output_path / f"{video_name}_frame{frame_idx}.jpg"
                cv2.imwrite(str(frame_filename), annotated_frame)

                # Write log
                log_file.write(f"[{datetime.now()}] {frame_filename.name}\n")
                for log_line in frame_log:
                    log_file.write(f"  - {log_line}\n")

                frame_idx += 1

            cap.release()

    print(f"Processing completed. Results saved in {output_path}")

def run_commands() -> None:
    """
    Run a predefined command to evaluate all videos using the process_videos script.
    """
    process_videos(
        input_folder="./datasets/Cam360",
        output_folder="./evals",
        model_path="./datasets/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt"
    )

if __name__ == "__main__":
    fire.Fire({
        'run_commands': run_commands
    })