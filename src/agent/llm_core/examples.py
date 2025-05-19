import yaml

config_path = "/home/chwenjun225/projects/KhaAnh/config/agent_config.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print(config["llm"]["type"])