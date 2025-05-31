from typing import Any, Dict, Optional
from ..base_agent import BaseAgent

from .image_sensor        import ImageSensor
from .image_preprocessor  import ImagePreprocessor
from .vision_extractor    import VisionExtractor
from .observation_filter  import ObservationFilter


class Perception(BaseAgent):
    """
    Vision-based Perception agent: captures an image, preprocesses it,
    obtains an LLM-based description, and filters the result.
    """
    def __init__(
            self,
            name: Optional[str] = "Perception",
            config_path: Optional[str] = None,
            image_source: str = None
        ):
        super().__init__(name=name, config_path=config_path)

        self.image_source = image_source    

        # TODO: Cần tổ chứuc lại các submodule này, để lưu các ảnh nhìn được vào bộ nhớ đệm trước 
        # Initialize submodules
        # self.sensor       = ImageSensor(source=image_source)
        # self.preprocessor = ImagePreprocessor()
        self.extractor    = VisionExtractor()
        self.filter       = ObservationFilter()

    def process(self) -> Dict[str, Any]:

        # TODO: Cần tìm cách triển khai nếu muổn realtime ta cần phải lưu các ảnh tìm được vào bộ nhớ đệm sau đó mới cho vào llama để inference 
        # try:
            # Capture image from camera or file_path
        #     raw_img = self.sensor.sense()
        # except Exception as e:
        #     return {"description": "", "error": str(e)}

        # try:
        #     # Encode image to base64 string 
        #     b64 = self.preprocessor.process(raw_img)
        # except Exception as e:
        #     return {"description": "", "error": str(e)}

        try:
            # Add system-prompt and user-prompt to memory and get description
            obs = self.extractor.extract(image_source=self.image_source, agent=self)
        except Exception as e: 
            return {"description": "", "error": str(e)}

        filtered = self.filter.apply(obs)
        # Log raw and filtered for debugging
        self.add_to_memory("assistant", f"Raw observation: {obs.get('description')}\nFiltered: {filtered}")
        return filtered