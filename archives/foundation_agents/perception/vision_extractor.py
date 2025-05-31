from typing import Any, Dict
from ..base_agent import BaseAgent


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
            # image_b64: str, # Base64-encoded JPEG string
            image_source: str = None
        ) -> Dict[str, Any]:
        """
        Args:
            agent: BaseAgent object to use for LLM interaction
            image_b64: Base64-encoded JPEG string

        Returns:
            A dict with keys: 'description' (str), 'raw' (full text)
        """

        # System prompt to switch to multi-modal model if needed
        agent.add_to_memory(role="system", content=("You are a helpful assistant"))

        # User prompt with the image data
        agent.add_to_memory(role="user", content=f"<|image|>Describe this image in detail, path image: {image_source}<|eot_id|>")

        # Generate with timeout
        description = agent.generate_response()
        return {"description": description.strip(), "raw": description}
