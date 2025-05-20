from typing import Any 
from ...base_agent import BaseAgent 


class Memory:
    """Encapsulates the agent's Memory component.

    Attributes:
        content: lưu trữ trạng thái memory hiện tại.

    Methods:
        update(): xây dựng prompt dựa trên action và observation, rồi gọi agent.generate_response().
    """

    def __init__(self, initial_content: Any = None):
        self.content = initial_content or ""

    def update(self, agent: BaseAgent, previous_action: Any, current_observation: Any) -> Any:
        # Build the user prompt
        prompt = (
            f"Previous memory:\n{self.content}\n\n"
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Update the agent's memory component accordingly."
        )

        # Push to agent's memory buffer and call LLM
        agent.add_to_memory("user", prompt)
        updated = agent.generate_response()

        # Store & return the new memory state
        self.content = updated.strip()
        return self.content