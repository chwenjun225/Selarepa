import os 
import cv2
import fire
import random 
import shutil
import numpy as np 
from PIL import Image 
from pathlib import Path 
from datetime import datetime
from glob import glob

from typing import List 
from typing import Tuple  
from typing import Dict 
from typing import Optional 

import torch 
from ultralytics import YOLO

from transformers import AutoModel 
from transformers import AutoTokenizer 
from transformers import AutoProcessor 
from transformers import PreTrainedTokenizer 
from transformers import PreTrainedModel 

# -------------------------------------- Inference with cam360 datasets --------------------------------------
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

def write_yolo_txt_with_labels(
        result,
        txt_filename: Path,
        class_labels: dict[int, str],
        use_label_str: bool = False, # False --> cho label bằng ID, True --> cho label bằng chữ 
        conf_threshold: float = 0.4
    ) -> None:
    """
    Ghi kết quả phát hiện từ YOLO vào file .txt.

    Args:
        result: Đối tượng kết quả từ YOLO (results[0]).
        txt_filename (Path): Đường dẫn file .txt để lưu.
        class_labels (dict): Mapping class_id → tên nhãn (str).
        use_label_str (bool): Nếu True thì ghi ra nhãn dạng string; nếu False thì ghi class_id số.
        conf_threshold (float): Ngưỡng confidence tối thiểu.
    """
    with open(txt_filename, "w") as f:
        for box in result.boxes:
            if box.conf[0] > conf_threshold:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                label = class_labels.get(cls, f"Unknown{cls}") if use_label_str else cls
                f.write(f"{label} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

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

    # Danh sách nhãn class
    class_labels = {
        0: "Person",            # 0: "Người",           0: "人 (Rén)", 
        1: "SafetyShoes",       # 1: "Giày bảo hộ",     1: "安全鞋 (Ānquán xié)",  
        2: "ESDSlippers",       # 2: "Dép tĩnh điện",   2: "防静电拖鞋 (Fáng jìngdiàn tuōxié)",
        3: "Head",              # 3: "Đầu",             3: "头部 (Tóubù)",
        4: "BlueUniform",       # 4: "Áo xanh",         4: "蓝色制服 (Lánsè zhìfú)",
        5: "WhiteUniform",      # 5: "Áo trắng",        5: "白色制服 (Báisè zhìfú)",  
        6: "BlackUniform",      # 6: "Áo đen",          6: "黑色制服 (Hēisè zhìfú)", 
        7: "OtherUniform",      # 7: "Áo Khác",         7: "其他制服 (Qítā zhìfú)",
        8: "Bending",           # 8: "Cúi",             8: "弯腰 (Wānyāo)",
        9: "FireExtinguisher",  # 9: "Bình Cứu Hỏa",    9: "灭火器 (Mièhuǒqì)",  
    }

    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))

        while True:
            ret, frame = cap.read()
            if not ret: break

            results = yolo(frame)
            if not results: continue
            result = results[0]

            # Save original frame
            orig_filename = original_path / f"{global_frame_index:06d}.jpg"
            cv2.imwrite(str(orig_filename), frame)

            # Save annotated frame
            annotated_frame = result.plot()
            yolo_filename = yolo_path / f"{global_frame_index:06d}.jpg"
            cv2.imwrite(str(yolo_filename), annotated_frame)

            # Save detection data to txt with readable label
            txt_filename = txt_path / f"{global_frame_index:06d}.txt"
            write_yolo_txt_with_labels(result, txt_filename, class_labels)

            global_frame_index += 1
        cap.release()

    print(f"Processing completed. Results saved in {output_path}")

def inference_cam360_dataset() -> None:
    """Run a predefined command to evaluate all videos using the process_videos script."""
    process_videos(
        input_folder=f"{os.getcwd()}/data/Cam360",
        output_folder=f"{os.getcwd()}/evals/Train_Fulian_25_04_20252",
        model_path=f"{os.getcwd()}/data/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt")
# -------------------------------------- Inference with cam360 datasets -------------------------------------- 



# -------------------------------------- Verify label is True or False -------------------------------------- 
def load_image(image_path: Path) -> torch.Tensor:
    """Load ảnh từ file và chuyển về tensor [3, H, W]."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1)

def read_yolo_labels(label_path: Path) -> List[str]:
    """Đọc file YOLO .txt và trả về danh sách các dòng nhãn."""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def crop_object(image: torch.Tensor, box: List[float]) -> Optional[Image.Image]:
    """Cắt object theo bbox, trả về ảnh PIL."""
    C, H, W = image.shape
    x1, y1, x2, y2 = map(int, box)
    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, W, H)

    cropped = image[:, y1:y2, x1:x2]
    if cropped.shape[1] == 0 or cropped.shape[2] == 0:
        return None
    return Image.fromarray(cropped.permute(1, 2, 0).cpu().numpy().astype("uint8"))

def expand_bbox(x1: int, y1: int, x2: int, y2: int, max_w: int, max_h: int, delta: int = 6) -> List[int]:
    """Giãn bounding box và đảm bảo không vượt ra ngoài ảnh."""
    x1 = max(0, x1 - delta // 2)
    y1 = max(0, y1 - delta // 2)
    x2 = min(max_w, x2 + delta // 2)
    y2 = min(max_h, y2 + delta // 2)
    return [x1, y1, x2, y2]

def verify_object_label_with_class_options(
        object_image: Image.Image,
        label_name: str,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        class_labels: Dict[int, str]
    ) -> Tuple[bool, Optional[str]]:
    """
    Xác minh object có đúng label hay không, và nếu sai thì gợi ý nhãn khác.

    Returns:
        - (True, None): nếu khớp với label_name.
        - (False, new_label): nếu model gợi ý nhãn khác hợp lệ.
        - (False, None): nếu model không gợi ý được gì hữu ích.
    """
    options = ', '.join(class_labels.values())
    prompt = f"Which of the following best describes the object in the image: {options}?"
    msgs = [{'role': 'user', 'content': [object_image, prompt]}]
    answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer).strip().lower()

    # Khớp với nhãn hiện tại
    if label_name.lower() == answer:
        return True, None

    # Tìm nhãn khác phù hợp
    for label in class_labels.values():
        if label.lower() in answer:
            return False, label

    return False, None

def run_relabel_pipeline(
    limit: int = 1000,
    data_dir: str = "./evals/Train_Fulian_25_04_20252",
    output_dir: str = "./verified_samples",
    model_name: str = "openbmb/MiniCPM-o-2_6"
) -> None:
    """Tiếp tục pipeline xác minh và sửa nhãn, bỏ qua các ảnh đã xử lý."""

    input_image_dir = Path(data_dir) / "original_frames"
    input_label_dir = Path(data_dir) / "txt_labels"
    output_image_dir = Path(output_dir) / "images"
    output_label_dir = Path(output_dir) / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=True
    ).to("cuda").eval()

    class_labels = {
        0: "Person", 1: "SafetyShoes", 2: "ESDSlippers", 3: "Head",
        4: "BlueUniform", 5: "WhiteUniform", 6: "BlackUniform",
        7: "OtherUniform", 8: "Bending", 9: "FireExtinguisher"
    }
    label2id = {v.lower(): k for k, v in class_labels.items()}

    # Đếm số lượng ảnh đã xử lý
    processed_files = set(f.stem for f in output_label_dir.glob("*.txt"))
    collected = len(processed_files)

    image_paths = sorted(input_image_dir.glob("*.jpg"))

    for img_path in image_paths:
        if collected >= limit:
            break

        if img_path.stem in processed_files:
            continue  # Skip đã xử lý

        label_path = input_label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        image_tensor = load_image(img_path)
        label_lines = read_yolo_labels(label_path)
        new_labels = []

        for line in label_lines:
            parts = line.split()
            if len(parts) != 6:
                continue
            cls_id = int(parts[0])
            label_name = class_labels.get(cls_id, "Unknown")
            box = list(map(float, parts[2:6]))
            obj_img = crop_object(image_tensor, box)
            if obj_img is None:
                continue

            match, predicted = verify_object_label_with_class_options(
                obj_img, label_name, tokenizer, model, class_labels
            )

            if match:
                new_labels.append(line)
            elif predicted and predicted.lower() in label2id:
                new_id = label2id[predicted.lower()]
                new_line = f"{new_id} {parts[1]} {' '.join(parts[2:])}"
                new_labels.append(new_line)

        if new_labels and new_labels != label_lines:
            shutil.copy(img_path, output_image_dir / img_path.name)
            with open(output_label_dir / label_path.name, "w") as f:
                for l in new_labels:
                    f.write(l + "\n")
            collected += 1

    print(f"🔁 Total collected: {collected} images (including previously processed) at {output_dir}")
# -------------------------------------- Verify label is True or False -------------------------------------- 



# -------------------------------------- Chuẩn hóa lại theo format yolov11 -------------------------------------- 
def convert_labels_to_yolo_format(
        label_dir: str = "./verified_samples/labels",  
        image_dir: str = "./verified_samples/images",  
    ) -> None:
    """
    Chuyển toàn bộ file label từ định dạng [class conf x1 y1 x2 y2] 
    -> sang YOLO format [class x_center y_center width height] (chuẩn hóa theo ảnh).

    Args:
        label_dir (str): Thư mục chứa file .txt label.
        image_dir (str): Thư mục chứa ảnh .jpg tương ứng.
    """
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)
    label_files = list(label_dir.glob("*.txt"))
    n_converted = 0

    for label_file in label_files:
        image_file = image_dir / (label_file.stem + ".jpg")
        if not image_file.exists():
            print(f"⚠️ Image file not found for {label_file.name}")
            continue

        img = cv2.imread(str(image_file))
        h, w = img.shape[:2]

        new_lines: List[str] = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue  # skip invalid lines
                cls_id, conf, x1, y1, x2, y2 = parts
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                new_line = f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                new_lines.append(new_line)

        with open(label_file, "w") as f:
            for line in new_lines:
                f.write(line + "\n")
        n_converted += 1

    print(f"✅ Converted {n_converted} label files to YOLOv8 training format.")
# -------------------------------------- Chuẩn hóa lại theo format yolov11 -------------------------------------- 



# -------------------------------------- Fine-tune & Eval mô hình YOLOv11 -------------------------------------- 
def check_verified_images(verified_data_dir: str, required: int = 1000) -> bool:
    image_dir = Path(verified_data_dir) / "images"
    count = len(list(image_dir.glob("*.jpg")))
    print(f"📸 Found {count} verified images.")
    return count >= required


def split_dataset(verified_dir: str, train_ratio: float = 0.8) -> tuple[Path, Path]:
    """
    Chia dữ liệu verified thành train/val theo tỷ lệ train_ratio.
    """
    images = sorted((Path(verified_dir) / "images").glob("*.jpg"))
    labels_dir = Path(verified_dir) / "labels"

    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)

    # Tạo thư mục
    train_img = Path(verified_dir) / "finetune/train/images"
    train_lbl = Path(verified_dir) / "finetune/train/labels"
    val_img = Path(verified_dir) / "finetune/val/images"
    val_lbl = Path(verified_dir) / "finetune/val/labels"

    for p in [train_img, train_lbl, val_img, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # Copy file
    for img in images[:split_idx]:
        lbl = labels_dir / f"{img.stem}.txt"
        shutil.copy(img, train_img / img.name)
        shutil.copy(lbl, train_lbl / lbl.name)

    for img in images[split_idx:]:
        lbl = labels_dir / f"{img.stem}.txt"
        shutil.copy(img, val_img / img.name)
        shutil.copy(lbl, val_lbl / lbl.name)

    return train_img.parent, val_img.parent


def generate_data_yaml(train_val_dir: Path) -> Path:
    """
    Tạo file data.yaml trỏ tới thư mục train/val đã chia.
    """
    yaml_path = train_val_dir.parent / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {train_val_dir.parent.resolve()}
train: train/images
val: val/images
names:
    0: Person
    1: SafetyShoes
    2: ESDSlippers
    3: Head
    4: BlueUniform
    5: WhiteUniform
    6: BlackUniform
    7: OtherUniform
    8: Bending
    9: FireExtinguisher
""")
    return yaml_path


def fine_tune_yolo(
    verified_data_dir: str = "./src/verified_samples",
    old_model_path: str = "./src/data/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt",
    save_dir: str = "./trained_new_yolo",
    epochs: int = 100,
    imgsz: int = 640
) -> Optional[Path]:
    """
    Fine-tune YOLOv11 từ tập ảnh đã xác thực.
    """
    if not check_verified_images(verified_data_dir):
        print("⛔ Not enough verified images to fine-tune.")
        return None

    train_val_dir, _ = split_dataset(verified_data_dir)
    data_yaml = generate_data_yaml(train_val_dir)

    model = YOLO(old_model_path)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        save=True,
        save_dir=save_dir,
        name="",
    )

    best_model = Path(save_dir) / "weights" / "best.pt"
    return best_model if best_model.exists() else None


def evaluate_model(model_path: str, data_yaml_path: str, imgsz: int = 640) -> float:
    """
    Đánh giá mô hình YOLO và trả về mAP50.
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path, imgsz=imgsz)
    return metrics.box.map50


def replace_model_if_better(
    old_model_path: str,
    new_model_path: str,
    eval_data_yaml: str
) -> None:
    """
    So sánh model mới và cũ theo mAP50 và thay thế nếu cần.
    """
    print("\n🔍 Evaluating current model...")
    old_map = evaluate_model(old_model_path, eval_data_yaml)

    print("\n🔍 Evaluating new model...")
    new_map = evaluate_model(new_model_path, eval_data_yaml)

    print(f"\n📊 mAP50 - old: {old_map:.4f} | new: {new_map:.4f}")

    if new_map > old_map:
        print("✅ New model is better. Replacing old model...")
        os.replace(new_model_path, old_model_path)
    else:
        print("⛔ Old model is better. Keeping original.")


def run_pipeline():
    """
    Pipeline:
    - Chia 80/20 dữ liệu xác thực
    - Fine-tune YOLOv11
    - So sánh và thay thế nếu model mới tốt hơn
    """
    verified_dir = "./src/verified_samples"
    eval_data_yaml = "./src/verified_samples/finetune/data.yaml"
    old_model = "./src/data/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt"

    best_model = fine_tune_yolo(
        verified_data_dir=verified_dir,
        old_model_path=old_model
    )

    if best_model is not None:
        replace_model_if_better(
            old_model_path=old_model,
            new_model_path=str(best_model),
            eval_data_yaml=eval_data_yaml
        )
# -------------------------------------- Fine-tune & Eval mô hình YOLOv11 -------------------------------------- 

if __name__ == "__main__":
    fire.Fire({
        # Inference with cam360 datasets
        "run_inference_cam360_dataset": inference_cam360_dataset, 

        # Verify label is True or False
        "relabel_objects": run_relabel_pipeline,

        # Chuẩn hóa lại format yolov11
        "convert_labels": convert_labels_to_yolo_format, 

        # Run pipeline finetune & evaluation
        "run_pipeline": run_pipeline,
    })

# TODO: Tìm hiểu các tham số và đánh giá xem mô hình nào tốt hơn 