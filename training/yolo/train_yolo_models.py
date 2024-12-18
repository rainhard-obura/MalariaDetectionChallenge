import yaml
from ultralytics import YOLO


def get_trained_yolo_models(config_files, dataset_path, device=0):
    trained_yolo_models = []
    for config_file in config_files:
        config_dict = dict(yaml.safe_load(open(config_file)))

        model = YOLO(config_dict.pop("model"))

        model.train(data=dataset_path, device=device, **config_dict)

        trained_yolo_models.append(model)

    # trained_yolo_models.append(YOLO('models/best.pt'))

    return trained_yolo_models
