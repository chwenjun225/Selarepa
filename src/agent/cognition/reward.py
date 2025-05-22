from typing import Any
from ..base_agent import BaseAgent

class Reward:
    """
    Encapsulates the agent's reward-signal component.

    Attributes:
        content: Internal representation of the agent's current reward signal.
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
        Build a prompt to generate or update the reward signal, send it through the agent's LLM,
        and store the updated result.

        Args:
            agent: The Cognition agent instance for LLM calls.
            previous_action: The action taken in the previous step.
            current_observation: The latest observation from the environment.

        Returns:
            The updated reward content.
        """
        prompt = (
            f"Previous reward signals:\n{self.content}\n\n"
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Produce the updated reward or learning signal."
        )
        agent.add_to_memory("user", prompt)
        updated_content = agent.generate_response().strip()
        self.content = updated_content
        return self.content