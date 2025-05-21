import base64
from typing import Any
import cv2

class ImagePreprocessor:
    """Converts raw BGR image data into a base64-encoded JPEG string."""

    def process(self, image: Any) -> str:

        # Encode as JPEG
        ret, buf = cv2.imencode('.jpg', image)
        if not ret:
            raise RuntimeError("Failed to encode image to JPEG")

        # Convert to base64
        jpg_bytes = buf.tobytes()
        b64 = base64.b64encode(jpg_bytes).decode('utf-8')
        return b64