import cv2
import fire
from pathlib import Path
from typing import List
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
    
    Save Directory:
        original_frames: Contain orginal image
        txt_labels: Contain label image
        yolo_drawn: Contain yolo drawn image 
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    yolo_path = output_path / "yolo_drawn"
    original_path = output_path / "original_frames"
    txt_path = output_path / "txt_labels"

    yolo_path.mkdir(parents=True, exist_ok=True)
    original_path.mkdir(parents=True, exist_ok=True)
    txt_path.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_path)
    video_files = collect_video_paths(str(input_path))

    global_frame_index = 0

    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame)
            if not results:
                continue
            result = results[0]

            # Save original frame
            orig_filename = original_path / f"{global_frame_index:06d}.jpg"
            cv2.imwrite(str(orig_filename), frame)

            # Save annotated frame
            annotated_frame = result.plot()
            yolo_filename = yolo_path / f"{global_frame_index:06d}.jpg"
            cv2.imwrite(str(yolo_filename), annotated_frame)

            # Save detection data to txt
            txt_filename = txt_path / f"{global_frame_index:06d}.txt"
            with open(txt_filename, "w") as f:
                for box in result.boxes:
                    if box.conf[0] > 0.4:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        f.write(f"{cls} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

            global_frame_index += 1

        cap.release()

    print(f"Processing completed. Results saved in {output_path}")

def run_commands() -> None:
    """
    Run a predefined command to evaluate all videos using the process_videos script.
    """
    process_videos(
        input_folder="./datasets/Cam360",
        output_folder="./evals/Train_Fulian_25_04_20252",
        model_path="./datasets/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt"
    )

if __name__ == "__main__":
    fire.Fire({
        'run_commands': run_commands
    })
