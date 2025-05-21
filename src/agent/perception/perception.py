from typing import Any, Dict, Optional
from ..base_agent import BaseAgent

from .perception_submodules.image_sensor        import ImageSensor
from .perception_submodules.image_preprocessor  import ImagePreprocessor
from .perception_submodules.vision_extractor    import VisionExtractor
from .perception_submodules.observation_filter  import ObservationFilter


class Perception(BaseAgent):
    """
    Vision-based Perception agent: captures an image, preprocesses it,
    obtains an LLM-based description, and filters the result.
    """
    def __init__(
            self,
            name: Optional[str] = "Perception",
            config_path: Optional[str] = None,
            image_source: str = "camera"
        ):
        super().__init__(name=name, config_path=config_path)

        # Initialize submodules
        self.sensor       = ImageSensor(source=image_source)
        self.preprocessor = ImagePreprocessor()
        self.extractor    = VisionExtractor()
        self.filter       = ObservationFilter()

    def process(self) -> Dict[str, Any]:
        try:
            # Capture image from camera or file_path
            raw_img = self.sensor.sense()
        except Exception as e:
            return {"description": "", "error": str(e)}

        try:
            # Encode image to base64 string 
            b64 = self.preprocessor.process(raw_img)
        except Exception as e:
            return {"description": "", "error": str(e)}

        try:
            obs = self.extractor.extract(self, b64)
        except Exception as e:
            return {"description": "", "error": str(e)}

        filtered = self.filter.apply(obs)
        # Log raw and filtered for debugging
        self.add_to_memory("assistant", f"Raw observation: {obs.get('description')}\nFiltered: {filtered}")
        return filtered