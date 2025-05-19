from typing import Dict, Any, Optional, Tuple
from agent.base import BaseAgent


class Cognition(BaseAgent):
    """Agent chịu trách nhiệm về nhận thức, cập nhật trạng thái tinh thần và suy luận hành động."""
    
    def __init__(self, config_path: Optional[str]=None) -> None:
        """Initialize the Cognition Agent.

        Args:
            config_path: Optional path to the configuration file
        """
        super().__init__("Cognition", config_path=config_path)
        
        # Base prompt for all cognition steps 
        self.base_prompt = (
            "You are the Cognition Agent, inspired by human brain cognitive architecture. "
            "Your update components of mental state and decide next actions."
        )

        self.add_to_memory("system", self.base_prompt)

    def memory(
            self, 
            previous_memory: Any, 
            previous_action: Any, 
            current_observation: Any
        ) -> Any:
        """
        Cập nhật thành phần Memory dựa trên memory và hành động tại 
        thời điểm t-1, và observation (quan sát) tại thời điểm t.
        """
        
        prompt = (
            f"Previous memory:\n{previous_memory}\n\n"
            f"Last action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Update the agent's memory component accordingly."
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()

    def world_model(
            self, 
            previous_world_model: Any, 
            previous_action: Any, 
            current_observation: Any, 
        ) -> Any:
        """
        Cập nhật World Model dựa trên world model và hành động tại 
        thời điểm t-1, và observation (quan sát) tại thời điểm t.
        """
        prompt = (
            f"Previous world model:\n{previous_world_model}\n\n"
            f"Last action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Update the internal representation of how the environment evolves."
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response() 
    
    def emotion(self, previous_emotion: Any, current_observation: Any) -> Any:
        """
        Cập nhật Emotion component dựa trên trạng thái cảm xúc tại
        thời điểm t-1, và observation tại thời điểm t.
        """
        prompt = (
            f"Previous emotion state:\n{previous_emotion}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Determine the updated emotional state (valence, arousal, etc.)."
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()

    def goal(self, previous_goal: Any, current_observation: Any) -> Any:
        """
        Cập nhật hoặc điều chỉnh Goal component dựa trên goal tại 
        thời điểm t-1, và observation tại thời điểm t.
        """
        prompt = (
            f"Previous goals/objective:\n{previous_goal}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Revise or propose new goals for the agent."
        )
        self.add_to_memory("user", prompt)
        return self.generate_response()

    def reward(self, previous_reward: Any, previous_action: Any, current_observation: Any) -> Any:
        """
        Generate tín hiệu Reward/Learning dựa trên reward, action 
        tại thời điểm t-1, và observation tại thời điểm t. 
        """
        prompt = (
            f"Previous reward signals:\n{previous_reward}\n\n"
            f"Last action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Produce the updated reward or learning signal."
        )
        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()

    def reasoning(self, mental_state: Dict[str, Any]) -> Any:
        """Suy luận hành động tiếp theo dựa trên toàn bộ trạng thái tinh thần."""
        prompt = (
            f"Current mental state:\n{mental_state}\n\n"
            "Decide the next action the agent should take."
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()

    def process(
        self,
        previous_state: Dict[str, Any],
        previous_action: Any,
        current_observation: Any
    ) -> Tuple[Dict[str, Any], Any]:
        """Thực hiện một chu kỳ nhận thức:
            1. Cập nhật memory, world_model, emotion, goal, reward
            2. Suy luận hành động tiếp theo

        Args:
            previous_state: dict chứa các thành phần của trạng thái tinh thần trước
            previous_action: hành động đã thực hiện ở bước trước
            current_observation: quan sát mới tại thời điểm t

        Returns:
            Tuple[updated_state, next_action]
        """
        new_memory      = self.memory(previous_state.get("memory"), previous_action, current_observation)
        new_world_model = self.world_model(previous_state.get("world_model"), previous_action, current_observation)
        new_emotion     = self.emotion(previous_state.get("emotion"), current_observation)
        new_reward      = self.reward(previous_state.get("reward"), previous_action, current_observation)
        new_goal        = self.goal(previous_state.get("goal"), current_observation)

        updated_state = {
            "memory":      new_memory,
            "world_model": new_world_model,
            "emotion":     new_emotion,
            "reward":      new_reward,
            "goal":        new_goal,
        }

        next_action = self.reasoning(updated_state)
        return updated_state, next_action
