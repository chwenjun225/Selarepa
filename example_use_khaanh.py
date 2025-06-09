import math
import fire
import numpy as np
import torch 
import matplotlib.pyplot as plt

from typing import List 
from PIL import Image
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from .lib.khaanh.modeling_navit_siglip import SiglipVisionConfig
from .lib.khaanh.modeling_navit_siglip import SiglipVisionTransformer
from .lib.khaanh.image_processing_khaanh import KhaanhImageProcessor


# Đường dẫn mặc định đến ảnh test
DEFAULT_IMAGE_PATH = "/home/chwenjun225/projects/Selarepa/data/Cam360/Inferenced_Train_Fulian_25_04_20252/original_frames/000000.jpg"

def load_image(path: str) -> Image.Image:
    """Tải ảnh RGB từ đường dẫn."""
    image = Image.open(path)
    return image.convert("RGB")


def plot_patches_grid(patches: list[list[Image.Image]], title_prefix: str = "Patch"):
    """Hiển thị lưới các patch ảnh."""
    rows, cols = len(patches), len(patches[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(patches[i][j])
            axes[i, j].set_title(f"{title_prefix} ({i},{j})", fontsize=8)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_image_list(images: list[Image.Image], cols: int = 4):
    """Hiển thị danh sách ảnh trên lưới."""
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else axes
    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].set_title(f"Slice {idx}", fontsize=10)
        axes[idx].axis("off")
    for i in range(len(images), len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def test_image_processing_khaanh_module(image_path: str = DEFAULT_IMAGE_PATH):
    image = load_image(image_path)
    processor = KhaanhImageProcessor()

    print("\n🎯 [1] find_best_resize")
    best_size = processor.find_best_resize(image.size, processor.scale_resolution, processor.patch_size)
    print(f"Best resize for {image.size} => {best_size}")

    print("\n🎯 [2] get_refine_size")
    grid = (3, 3)
    refined_size = processor.get_refine_size(image.size, grid, processor.scale_resolution, processor.patch_size)
    print(f"Refined size for grid {grid}: {refined_size}")

    print("\n🎯 [3] split_to_patches")
    resized_image = image.resize(refined_size)
    patches = processor.split_to_patches(resized_image, grid)
    print(f"Split into {len(patches)} rows x {len(patches[0])} cols")
    plot_patches_grid(patches)

    print("\n🎯 [4] get_sliced_images")
    sliced_images = processor.get_sliced_images(image)
    print(f"Total slices (incl. source): {len(sliced_images)}")
    plot_image_list(sliced_images)

    print("\n🎯 [5] get_sliced_grid")
    best_grid = processor.get_sliced_grid(image.size, processor.max_slice_nums)
    print(f"Best sliced grid: {best_grid}")

    print("\n🎯 [6] slice_image")
    src_image, patch_grid, sliced_grid = processor.slice_image(image)
    print(f"Resized source image: {src_image.size}, Grid: {sliced_grid}, Total patches: {sum(len(row) for row in patch_grid)}")

    print("\n🎯 [7] get_slice_image_placeholder")
    placeholder = processor.get_slice_image_placeholder(image.size, image_idx=0)
    print("--- Placeholder Output ---")
    print(placeholder)

    print("\n🎯 [8] reshape_by_patch")
    img_np = np.asarray(image.resize((224, 224))).astype(np.float32) / 255
    reshaped = processor.reshape_by_patch(img_np.transpose(2, 0, 1))
    print(f"Reshaped patch shape: {reshaped.shape}")

    print("\n🎯 [9] preprocess")
    batch = processor.preprocess([image], return_tensors="pt")
    print(f"Output pixel_values: {len(batch['pixel_values'][0])} patches")
    print(f"Image sizes: {batch['image_sizes']}")
    print(f"Target patch sizes: {batch['tgt_sizes']}")
    for idx, tensor in enumerate(batch["pixel_values"][0]):
        print(f"Patch Tensor {idx}: {tensor.shape}")


if __name__ == "__main__":
    fire.Fire({
        "img_proc_module": test_image_processing_khaanh_module,
        "visual_encoder": test_siglip_visual_encoder_module, # TODO: Cần xem lại cách minicpm  # type: ignore
    })
