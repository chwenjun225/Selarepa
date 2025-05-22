from typing import Any
from ..base_agent import BaseAgent

class Goal:
    """
    Encapsulates the agent's goal-management component.

    Attributes:
        content: Internal representation of the agent's current goals.
    """

    def __init__(self, initial_content: Any = None):
        self.content = initial_content or ""

    def update(
        self,
        agent: BaseAgent,
        previous_action: Any,
        current_observation: Any
    ) -> Any:
        """
        Build a prompt to revise or propose goals, send it through the agent's LLM, and store the updated result.

        Args:
            agent: The Cognition agent instance for LLM calls.
            previous_action: The action taken in the previous step.
            current_observation: The latest observation from the environment.

        Returns:
            The updated goal content.
        """
        prompt = (
            f"Previous goals:\n{self.content}\n\n"
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Revise or propose new goals for the agent."
        )
        agent.add_to_memory("user", prompt)
        updated_content = agent.generate_response().strip()
        self.content = updated_content
        return self.content
