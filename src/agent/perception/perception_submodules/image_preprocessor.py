import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

class ImagePreprocessor:
    """Converts raw BGR image data into a base64-encoded jpg string."""

    def process(self, bgr_image: Image.Image) -> str:
        pil_image = cv2_to_pil(bgr_image)
        pil_image_resized = pil_image.resize((640, 480))
        b64 = image_to_base64(pil_image_resized, format="JPEG")
        return b64 

def image_to_base64(pil_img: Image.Image, format="JPEG") -> str:
    """Convert PIL image to base64 string with proper MIME type prefix."""
    with BytesIO() as buffer:
        pil_img.save(buffer, format=format)
        b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        mime_type = "jpeg" if format.upper() == "JPEG" else format.lower()
        return f"data:image/{mime_type};base64,{b64_image}"

def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
