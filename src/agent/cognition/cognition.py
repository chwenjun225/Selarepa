from typing import Dict, Any, Optional, Tuple 
from ..base_agent import BaseAgent 
from .cognition_submodules.memory import Memory 

# BaseAgent vốn đã có sẵn bộ nhớ của nó rồi, vậy khi ta tạo ra module memory, vậy mục đích của module này nhằm chứa các phương thức giúp quản lý memory từ Cognition 
class Cognition(BaseAgent):
    """Agent responsible for cognition, mental state updating, and action reasoning."""

    def __init__(self, name: Optional[str] = "Cognition", config_path: Optional[str]=None) -> None:
        """Initialize the Cognition Agent.

        Args:
            config_path: Optional path to the configuration file
        """
        super().__init__(name=name, config_path=config_path)

        # General system prompt for all cognition steps 
        self.base_prompt = ( # TODO: Need optimize this prompt
            "You are the Cognition Agent, inspired by human brain cognitive architecture. "
            "You update components of mental state and decide next actions."
        )

        # Add system prompt to agent memory 
        self.add_to_memory("system", self.base_prompt) # TODO: This Memory need to be update with cognition_submodules/memory.py  

    def memory(self, previous_memory: Any, previous_action: Any, current_observation: Any) -> Any:
        """Cập nhật thành phần Memory dựa trên memory và action tại thời điểm t-1, và observation tại thời điểm t."""

        prompt = ( # TODO: Optimize this prompt 
            f"Previous memory:\n{previous_memory}\n\n" #TODO: Cần làm rõ hơn phần này previous memory include (world_model, memory, emotional, goal, reward)
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Update the agent's memory component accordingly."
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()

    def world_model(self, previous_world_model: Any, previous_action: Any, current_observation: Any) -> Any:
        """Cập nhật World Model dựa trên world model và action tại thời điểm t-1, và observation tại thời điểm t"""
        prompt = (
            f"Previous world model:\n{previous_world_model}" # TODO: Thành phần previous_emotion cần phải được lấy từ cognition_submodules/world_model.py
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Update the internal representation of how the environment evolves." # TODO: Xác minh lại chỗ này xem có cần reasoning không, nếu có hãy sử dụng deepseek-r1
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response() 
    
    def emotion(self, previous_emotion: Any, current_observation: Any) -> Any:
        """Cập nhật Emotion dựa trên emotion tại thời điểm t-1, và observation tại thời điểm t."""
        prompt = (
            f"Previous emotion state:\n{previous_emotion}\n\n" # TODO: Thành phần previous_emotion cần phải được lấy từ cognition_submodules/emotion.py
            f"Current observation:\n{current_observation}\n\n"
            "Determine the updated emotional state (valence, arousal, etc)."
        )

        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()
    
    def goal(self, previous_goal: Any, current_observation: Any) -> Any:
        """Cập nhật hoặc điều chỉnh Goal dựa trên goal tại thời điểm t-1, và observation tại thời điểm t."""
        prompt = (
            f"Previous goals:\n{previous_goal}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Revise or propose new goals for the agent."
        )
        self.add_to_memory("user", prompt)
        return self.generate_response()
    
    def reward(self, previous_reward:Any, previous_action: Any, current_observation: Any) -> Any:
        """Generate tín hiệu Reward/Learning dựa trên reward, action tại thời điểm t-1, và observation tại thời điểm t."""
        prompt = (
            f"Previous reward signals:\n{previous_reward}\n\n"
            f"Previous action taken:\n{previous_action}\n\n"
            f"Current observation:\n{current_observation}\n\n"
            "Produce the updated reward or learning signal."
        )
        # Add user prompt to memory 
        self.add_to_memory("user", prompt)
        return self.generate_response()
    
    def reasoning(self, mental_state: Dict[str, Any]) -> Any:
        """Suy luận đưa ra hành động tiếp theo dựa trên toàn bộ trạng thái thần kinh của agent."""
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
        ) -> Tuple[Dict[str, Any]]:
        """Thực hiện một chu kỳ cognition:
            1. Cập nhật memory, world_model, emotion, goal, reward 
            2. Suy luận đưa ra hành động tiếp theo 

        Args:
            previous_state: dictionaries chứa các thành phần mental state ở thời điểm t-1
            previous_action: action đã thực hiện ở thời điểm t-1
            current_observation: observation tại thời điểm t

        Returns:
            Tuple[updated_state, next_action]
        """

        # Update Cognition's submodules
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

        # From updated state, predict the next action, it's could be internal action (planning, decision-making) or external action (e.g. open the box, etc)
        next_action = self.reasoning(updated_state)
        return updated_state, next_action
