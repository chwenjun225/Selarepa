import os 
import cv2
import fire
import shutil
import numpy as np 
from PIL import Image 
from pathlib import Path 
from datetime import datetime

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
# -------------------------------------- Self-Label-Repair-System -------------------------------------- 
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
    output_dir: str = "./relabel_data",
    model_name: str = "openbmb/MiniCPM-o-2_6"
) -> None:
    """Pipeline xác minh nhãn từng object và sửa nếu sai, lưu vào thư mục riêng."""

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

    image_paths = sorted(input_image_dir.glob("*.jpg"))
    collected = 0

    for img_path in image_paths:
        if collected >= limit:
            break

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

    print(f"🔁 Collected {collected} relabeled images at {output_dir}")


# -------------------------------------- Self-Label-Repair-System -------------------------------------- 
# -------------------------------------- Fine-tune mô hình YOLO mới từ các ảnh đã xác thực -------------------------------------- 


def fine_tune_yolo_on_verified_data(
    verified_data_dir: str = "./verified_samples",
    old_model_path: str = "./data/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt",
    save_dir: str = "./trained_new_yolo",
    epochs: int = 20,
    imgsz: int = 640
) -> Path:
    """
    Fine-tune lại YOLO từ model cũ bằng tập dữ liệu đã xác thực.

    Args:
        verified_data_dir (str): Thư mục chứa "images" và "labels" từ bước xác thực.
        old_model_path (str): Trọng số mô hình YOLO cũ.
        save_dir (str): Nơi lưu kết quả training mới.
        epochs (int): Số epoch để huấn luyện.
        imgsz (int): Kích thước ảnh khi train.

    Returns:
        Path: Đường dẫn đến mô hình tốt nhất sau fine-tune.
    """
    data_yaml_path = Path(verified_data_dir) / "data.yaml"

    # Tạo file data.yaml cho YOLO
    with open(data_yaml_path, "w") as f:
        f.write(f"""
path: {verified_data_dir}
train: images
val: images

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

    # Fine-tune YOLO
    model = YOLO(old_model_path)
    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        save=True,
        save_dir=save_dir,
        project=None,
        name=None
    )
    return Path(save_dir) / "weights" / "best.pt"


def evaluate_model(
    model_path: str,
    data_dir: str = "./evals/Train_Fulian_25_04_20252",
    imgsz: int = 640
) -> float:
    """
    Đánh giá mAP của một mô hình YOLO trên tập dữ liệu định trước.

    Args:
        model_path (str): Đường dẫn đến model YOLO định dạng .pt
        data_dir (str): Thư mục chứa ảnh và nhãn (original_frames + txt_labels)
        imgsz (int): Kích thước ảnh resize khi đánh giá

    Returns:
        float: Giá trị mAP50 thu được
    """
    # Tạo file YAML mô tả dữ liệu tạm thời
    yaml_path = Path(data_dir) / "eval_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""
path: {data_dir}
train: original_frames
val: original_frames
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

    # Load model và thực hiện đánh giá
    model = YOLO(model_path)
    metrics = model.val(data=str(yaml_path), imgsz=imgsz, split="val", save=False)
    return metrics.box.map50


def replace_model_if_better(
    old_model_path: str,
    new_model_path: str,
    eval_data_dir: str = "./evals/Train_Fulian_25_04_20252"
) -> None:
    """
    So sánh model mới và model cũ, nếu model mới tốt hơn (mAP cao hơn) thì thay thế.

    Args:
        old_model_path (str): Trọng số mô hình đang dùng
        new_model_path (str): Trọng số mô hình mới huấn luyện
        eval_data_dir (str): Thư mục dữ liệu đánh giá (ảnh + label)
    """
    print("\n🔍 Evaluating old model...")
    old_map = evaluate_model(old_model_path, data_dir=eval_data_dir)

    print("\n🔍 Evaluating new model...")
    new_map = evaluate_model(new_model_path, data_dir=eval_data_dir)

    print(f"\n📊 mAP50 - old: {old_map:.4f} | new: {new_map:.4f}")

    if new_map > old_map:
        print("✅ New model is better. Replacing old model...")
        os.replace(new_model_path, old_model_path)
    else:
        print("⛔ Old model is better. Keeping original model.")


def run_self_label_repair_system():
    """
    Chạy toàn bộ pipeline self-label-repair:
    1. Xác thực lại nhãn bằng MMLLM (MiniCPM)
    2. Fine-tune YOLO trên ảnh đã xác thực
    3. So sánh hiệu quả và thay thế model nếu cần
    """
    from self_label_repair import run_verification_pipeline, fine_tune_yolo_on_verified_data

    run_verification_pipeline()
    best_new_model = fine_tune_yolo_on_verified_data()

    replace_model_if_better(
        old_model_path="./data/Cam360/Weight/Train_Fulian_25_04_20252/weights/best.pt",
        new_model_path=str(best_new_model)
    )

# -------------------------------------- Fine-tune mô hình YOLO mới từ các ảnh đã xác thực -------------------------------------- 


if __name__ == "__main__":
    fire.Fire({
        "run_inference_cam360_dataset": inference_cam360_dataset, 
        "relabel_objects": run_relabel_pipeline,
    })
