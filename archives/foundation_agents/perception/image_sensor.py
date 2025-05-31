import numpy as np 
import cv2 
from PIL import Image

class ImageSensor:
    """
    Captures an image from a webcam or loads from file.

    Attributes:
        source: "camera" or file path
    """
    def __init__(self, source: str = "camera") -> None:  # or file path
        self.source = source

    def sense(self) -> np.ndarray:
        """
        Returns raw image data (BGR numpy array).
        """
        if self.source == "camera":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera")
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Failed to capture image from camera")
            return frame
        else:
            # Load from file
            frame = cv2.imread(self.source)
            if frame is None:
                raise FileNotFoundError(f"Image file not found: {self.source}")
            return frame
