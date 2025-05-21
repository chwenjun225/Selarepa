import fire 
from agent.cognition.cognition import Cognition


def main():
    # Initialize the Cognition agent
    agent = Cognition(
        config_path="config/agent_config.yaml", 
        name="Cognition"
    )

    # Initial mental state 
    # TODO: Mỗi thành phần như memory, world_model, emotion, reward, goal cần phải là một class, 
    # chứa các thuộc tính riêng mô phỏng hệ thống thần kinh nhận thức 
    previous_mental_state = {
        "memory": "",            # empty or default memory
        "world_model": "",       # empty or default world model
        "emotion": "neutral",    # e.g., neutral emotional state
        "reward": "",            # empty or default reward signal
        "goal": "explore"        # e.g., initial goal: explore
    }

    # Initial the previous action (None for the first cycle)
    previous_action = None

    # Provide a new observation from the environment
    current_observation = (
        "You see a man sitting in front of the computer, "
        "he can be your dad, he created you."
    )

    # Run one cognition cycle: update mental state and infer next action
    updated_state, next_action = agent.process(
        previous_mental_state,
        previous_action,
        current_observation, 
    )

    # Print out the updated mental state components
    print("=== Updated Mental State ===")
    for component, value in updated_state.items():
        print(f"{component}: {value}\n")

    # Print out the next action decided by the agent
    print("=== Next Action ===")
    print(next_action)


if __name__ == "__main__":
    fire.Fire(main)


"""
không cần phải build từng submodule con trong cognition, ta đã có lớp base agent rồi, ta sẽ biến các submodule của bộ nhớ này thành các attribute hoặc method nhỉ 



















"""
