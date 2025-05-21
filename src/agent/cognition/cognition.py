from typing import Any, Dict, Optional, Tuple
from ..base_agent import BaseAgent

from .cognition_submodules.memory       import Memory
from .cognition_submodules.world_model  import WorldModel
from .cognition_submodules.emotion      import Emotion
from .cognition_submodules.goal         import Goal
from .cognition_submodules.reward       import Reward
from .cognition_submodules.reasoning    import Reasoning


class Cognition(BaseAgent):
    """
    Agent responsible for cognition, updating mental 
    state components and reasoning actions.
    """

    def __init__(
        self,
        name: Optional[str] = "Cognition",
        config_path: Optional[str] = None
    ) -> None:
        super().__init__(name=name, config_path=config_path)

        # Base system prompt
        self.base_prompt = (
            "You are the Cognition Agent, inspired by human brain cognitive architecture. "
            "You update components of mental state and decide next actions."
        )
        self.add_to_memory("system", self.base_prompt)

        # Initialize submodules
        self.memory_module      = Memory()
        self.world_model_module = WorldModel()
        self.emotion_module     = Emotion()
        self.goal_module        = Goal()
        self.reward_module      = Reward()

    def process(
        self,
        previous_state: Dict[str, Any],
        previous_action: Any,
        current_observation: Any
    ) -> Tuple[Dict[str, Any], Any]:

        # Learning step: sync and update memory submodule
        # Ensure external previous memory is used as starting point
        self.memory_module.content = previous_state.get("memory", self.memory_module.content)
        new_memory      = self.memory_module.update(self, previous_action, current_observation)

        # Other submodules (use their own internal content)
        new_world_model = self.world_model_module.update(self, previous_action, current_observation)
        new_emotion     = self.emotion_module.update(self, previous_action, current_observation)
        new_reward      = self.reward_module.update(self, previous_action, current_observation)
        new_goal        = self.goal_module.update(self, previous_action, current_observation)

        updated_state = {
            "memory":      new_memory,
            "world_model": new_world_model,
            "emotion":     new_emotion,
            "reward":      new_reward,
            "goal":        new_goal,
        }

        # Reasoning step
        next_action = Reasoning.decide(self, updated_state)
        return updated_state, next_action
