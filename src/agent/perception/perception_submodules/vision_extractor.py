from typing import Any, Dict
from ...base_agent import BaseAgent

class VisionExtractor:
    """
    Sends base64 image data to a vision-enabled LLM and retrieves a description.
    """
    def __init__(
            self,
            model_name: str = "llama3.2-11b-vision-instruct",
            timeout_s: int = 30
        ) -> None:
        self.model_name = model_name
        self.timeout_s = timeout_s

    def extract(self,
                agent: BaseAgent,
                image_b64: str) -> Dict[str, Any]:
        """
        Args:
            agent: BaseAgent configured for llama3.2-11b-vision-instruct
            image_b64: Base64-encoded JPEG string
        Returns:
            A dict with keys: 'description' (str), 'raw' (full text)
        """
        # System prompt to switch to multi-modal model if needed
        agent.add_to_memory(
            "system",
            f"Use model {self.model_name} to describe the contents of an image."
        )
        # User prompt with image data
        prompt = (
            "Here is an image (base64-encoded JPEG). "
            "Please describe in detail what you see, including objects, colors, and context:\n"
            f"<IMAGE_DATA>\n{image_b64}\n</IMAGE_DATA>"
        )
        agent.add_to_memory("user", prompt)
        # Generate with timeout
        description = agent.generate_response()
        return {"description": description.strip(), "raw": description}