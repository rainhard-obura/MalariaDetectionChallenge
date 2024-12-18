import logging

import yaml


def load_yaml_config(filepath):
    logging.info(f"Loading YAML config from {filepath}")
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    logging.info(f"YAML config loaded successfully from {filepath}")
    return config


def create_structured_config(wandb_config):
    logging.info("Creating structured config from wandb parameters")
    config = {
        "input": {},
        "postprocessing": {
            "ensemble_ttayolo": {},
            "ensemble_ttadetr": {},
            "individual_detr": {},
            "individual_yolo11": {},
            "ensemble_all": {},
        },
    }

    # Map wandb parameters to config structure
    for key, value in wandb_config.items():
        if key.startswith("input_"):
            config["input"][key[6:]] = value  # remove 'input_' prefix
        elif key.startswith("postprocessing_"):
            parts = key.split("_")[1:]  # Remove 'postprocessing_' prefix

            if len(parts) >= 3:
                section = "_".join(parts[:2])
                param = "_".join(parts[2:])

                # Handle class-specific parameters
                if "Trophozoite" in param or "WBC" in param:
                    class_type, param_name = param.split("_", 1)
                    if section in config["postprocessing"]:
                        if (
                            f"{class_type}_params"
                            not in config["postprocessing"][section]
                        ):
                            config["postprocessing"][section][
                                f"{class_type}_params"
                            ] = {}

                        config["postprocessing"][section][f"{class_type}_params"][
                            param_name
                        ] = value
                else:
                    if section in config["postprocessing"]:
                        config["postprocessing"][section][param] = value

        # Map output path
        elif key == "output_path":
            config["output_path"] = value

    logging.info("Structured config created successfully")
    return config
