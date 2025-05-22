from typing import Any 
from ..base_agent import BaseAgent 


class Memory:
    """
    Encapsulates the agent's cognitive memory component.

    Attributes:
        content: The internal representation of the agent's memory (e.g., summary string, list of facts).
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
        Build a prompt to update the memory component, send it through the agent's LLM,
        and store the updated result.

        Args:
            agent: The Cognition agent instance, used for LLM calls.
            previous_action: The action taken in the previous step.
            current_observation: The latest observation from the environment.

        Returns:
            The updated memory content.
        """
        # Construct the user prompt
        prompt = (
            f"Previous memory:\n{self.content}\n\n"
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Update the agent's memory component accordingly."
        )

        # Append the prompt to the agent's chat history and call the LLM
        agent.add_to_memory("user", prompt)
        updated_content = agent.generate_response().strip()

        # Save and return the new memory
        self.content = updated_content
        return self.content