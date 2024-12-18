import os
import random

import numpy as np
import torch
import yaml

from .core.model import get_model, get_processor
from .core.train import train


def make_deterministic(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_trained_detr_models(config_files, dataset_path) -> list:
    make_deterministic()
    trained_detr_models = []
    for config_file in config_files:
        config_dict = dict(yaml.safe_load(open(config_file)))
        model = get_model(config_dict.pop("model"))
        processor = get_processor(config_dict.pop("processor"))
        device = config_dict["device"]
        train(config_dict, model, processor, data_dir=dataset_path, device=device)
        os.makedirs("models", exist_ok=True)

        torch.save(model.state_dict(), f"models/detr_model.pth")

        # Load pretrained weights instead of training
        # model.load_state_dict(torch.load('models/detr_model_1.pth')) # MAKE SURE THE LOADED MODEL WAS TRAINED WITH THE SAME CONFIG
        # model_path = 'models/detr_model_1.pth'
        # model.load_state_dict(torch.load(model_path, map_location='cpu'))
        # model.to('cuda:0')
        trained_detr_models.append(model)
    return trained_detr_models
