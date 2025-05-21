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
            role="system",
            content=("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|>""")
        )

        # User prompt with the image data
        agent.add_to_memory(
            role="user", 
            content=f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>Describe this image: {image_b64}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )

        # Generate with timeout
        description = agent.generate_response()
        return {"description": description.strip(), "raw": description}
