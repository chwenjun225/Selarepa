from agent.perception.perception import Perception 


def main() -> None:
    # Set the image source to a file path or 'camera' for live capture
    image_source = "/home/chwenjun225/projects/KhaAnh/assets/dieu_binh_nu.jpg"

    # Use 'camera' or provide a filepath: e.g. 'test.jpg'
    perc = Perception(
        config_path="config/agent_config.yaml", 
        image_source=image_source, 
    )

    # Invoke the perception process
    obs = perc.process()

    if obs.get("error"):
        print("Error during perception:", obs["error"])
    else:
        print("=== Observation ===")
        print(obs["description"])

if __name__ == "__main__":
    main()
