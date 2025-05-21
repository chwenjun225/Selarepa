from typing import Any, Dict
from ...base_agent import BaseAgent


class VisionExtractor:
    """Sends base64 image data to a vision-enabled LLM and retrieves a description."""
    def __init__(
            self,
            timeout_s: int = 30
        ) -> None:

        # Set timeout for LLM response if it take too long
        self.timeout_s = timeout_s

    def extract(
            self,
            agent: BaseAgent,
            image_b64: str
        ) -> Dict[str, Any]:
        """
        Args:
            agent: BaseAgent object to use for LLM interaction
            image_b64: Base64-encoded JPEG string
        Returns:
            A dict with keys: 'description' (str), 'raw' (full text)
        """

        # System prompt to switch to multi-modal model if needed
        agent.add_to_memory(
            "system",
            "Describe the contents of an image."
        )

        # User prompt with the image data
        prompt = (
            "You are a vision-enabled assistant. I will provide an image encoded in base64 format. "
            "Your task is to look at the image and provide a detailed description of what you see. "
            "Please describe the main objects, their colors, relative positions, background, actions (if any), "
            "and the overall context of the scene.\n\n"
            "Be as descriptive as possible, and include any notable attributes or relationships between objects.\n\n"
            "Here is the image:\n"
            f"<IMAGE>\n{image_b64}\n</IMAGE>"
        )
        agent.add_to_memory("user", prompt)

        # Generate with timeout
        description = agent.generate_response()
        return {"description": description.strip(), "raw": description}
