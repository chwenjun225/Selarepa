from typing import Dict, Any

class ObservationFilter:
    """Simple length-based filter on the LLM description."""
    
    def __init__(self, min_chars: int = 10):
        self.min_chars = min_chars

    def apply(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if len(obs.get("description", "")) < self.min_chars:
            return {"description": "", "raw": obs.get("raw", "")}
        return obs