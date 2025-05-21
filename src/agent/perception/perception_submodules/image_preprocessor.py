import base64
from io import BytesIO
from PIL import Image

class ImagePreprocessor:
    """Converts raw BGR image data into a base64-encoded JPEG string."""

    def process(self, image: Image.Image) -> str:
        b64 = image_to_base64(image.resize((640, 480))) # TODO: Need to debug this image 
        return b64
# TODO: Có vẻ ảnh bị sai, cần debug lại
# (Research) (base) chwenjun225@chwenjun225:~/projects/KhaAnh$ python -m examples.example_perception_usage
# 2025-05-21 20:25:06,373 - httpx - INFO - HTTP Request: POST http://localhost:11434/v1/chat/completions "HTTP/1.1 200 OK"
# === Observation ===
# The image is a cartoon of a smiling cat sitting on a windowsill, looking out at the city. The cat has a thought bubble above its head with a speech bubble that says "I wish I could go outside and explore." The background is a blurred image of a cityscape with tall buildings and cars driving by.

# Here's a more detailed description:

# *   The cartoon style is reminiscent of classic cartoons from the 1940s and 1950s.
# *   The cat's facial expression is one of longing, with its eyes looking wistfully out at the city.
# *   The thought bubble above the cat's head contains a speech bubble that says "I wish I could go outside and explore."
# *   The background is a blurred image of a cityscape, with tall buildings and cars driving by.

def image_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL or NumPy image to base64 string (PNG format), optimized for real-time usage."""
    with BytesIO() as buffer:
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
