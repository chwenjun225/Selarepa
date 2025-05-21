from perception import Perception

def main():
    # Use 'camera' or provide a filepath: e.g. 'test.jpg'
    perc = Perception(config_path="config/agent_config.yaml", image_source="camera")
    obs = perc.process()
    if obs.get("error"):
        print("Error during perception:", obs["error"])
    else:
        print("=== Observation ===")
        print(obs["description"])

if __name__ == "__main__":
    main()