import glob
import os
import random
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from ema_pytorch import EMA
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.utils.data import DataLoader
from transformers import BatchFeature

from .datasets.detr_dataset import MalariaDataset, Collator
from .infer import ImagePrediction
from .soap import SOAP
from .transforms import train_augment_and_transform


def convert_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (xmin, ymin, xmax, ymax) to (x, y, width, height) format."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(
    predictions: List[ImagePrediction],
) -> List[Dict[str, Union[int, List[float], float]]]:
    """Prepare predictions for COCO detection format."""
    coco_results = []
    for prediction in predictions:
        for pred in prediction.predictions:
            coco_results.append(
                {
                    "image_id": prediction.img_name,
                    "category_id": pred.class_name,
                    "bbox": list(pred.bbox),
                    "score": pred.confidence,
                }
            )
    return coco_results


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


def move_to_device(
    batch: Union[Dict, List, Tuple, BatchFeature, torch.Tensor], device: torch.device
) -> Union[Dict, List, Tuple, BatchFeature, torch.Tensor]:
    """Move all tensors in a batch to the specified device."""
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(item, device) for item in batch)
    elif isinstance(batch, BatchFeature):
        return BatchFeature(
            {key: move_to_device(value, device) for key, value in batch.items()}
        )
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


def convert_bbox_format(
    bbox: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    """Convert bounding box format from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height)."""
    xmin, ymin, xmax, ymax = bbox
    return xmin, ymin, xmax - xmin, ymax - ymin


def denormalize_bboxes(
    normalized_bboxes: List[List[float]], image_width: int, image_height: int
) -> List[List[float]]:
    """Convert normalized bounding boxes to absolute pixel values."""
    return [
        [
            x_center * image_width - (box_width * image_width / 2),
            y_center * image_height - (box_height * image_height / 2),
            x_center * image_width + (box_width * image_width / 2),
            y_center * image_height + (box_height * image_height / 2),
        ]
        for x_center, y_center, box_width, box_height in normalized_bboxes
    ]


def add_missing_image_ids(
    predictions: pd.DataFrame, all_ids: List[str]
) -> pd.DataFrame:
    """Add missing Image_IDs to the predictions DataFrame."""
    # Get unique image IDs from the predictions
    predicted_ids = set(predictions["Image_ID"].unique())

    # Find missing IDs
    missing_ids = set(all_ids) - predicted_ids

    # Create DataFrame for missing IDs
    missing_df = pd.DataFrame(
        {
            "Image_ID": list(missing_ids),
            "class": "NEG",
            "confidence": 0.0,
            "ymin": 0.0,
            "xmin": 0.0,
            "ymax": 0.0,
            "xmax": 0.0,
        }
    )

    # Concatenate original predictions with missing IDs
    result_df = pd.concat([predictions, missing_df], ignore_index=True)

    # Sort the DataFrame by img_name
    result_df = result_df.sort_values("Image_ID").reset_index(drop=True)

    return result_df


def get_device():
    # Check if the SWEEP_GPU environment variable is set
    gpu_id = os.getenv("DEVICE", "0")  # Default to GPU 0 if not set
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print("device", device)
    return device


def initialize_optimizer(
    model, config, n_steps
) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler
]:
    # Create optimizer
    optimizer = SOAP(model.parameters(), lr=config["training"]["learning_rate"])

    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=config["training"]["warmup_steps"],
    )

    # Create cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=n_steps - config["training"]["warmup_steps"]
    )

    # Combine warmup and cosine schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config["training"]["warmup_steps"]],
    )

    # Initialize EMA
    ema = None
    if config["ema"]["enabled"]:
        ema = EMA(
            model,
            beta=config["ema"]["beta"],
            update_after_step=config["ema"]["update_after_step"],
            update_every=config["ema"]["update_every"],
        )

    return optimizer, scheduler, ema


def get_train_dataloader(config, data_dir, processor):
    # Load YAML config
    with open(data_dir, "r") as f:
        dataset_config = yaml.safe_load(f)

    # Get base path and train folder
    base_path = dataset_config["path"]
    train_path = os.path.join(base_path, dataset_config["train"])

    # Get class names mapping
    class_names = dataset_config["names"]

    # Initialize lists for building DataFrame
    image_ids = []
    classes = []
    boxes = []

    # Read all label files
    label_files = glob.glob(os.path.join(train_path, "labels", "*.txt"))
    for label_file in label_files:
        img_id = os.path.basename(label_file).replace(".txt", ".jpg")
        img_path = os.path.join(train_path, "images", img_id)

        # Get image dimensions for denormalization
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        with open(label_file, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(
                    float, line.strip().split()
                )

                # Convert YOLO format to x1,y1,x2,y2 and denormalize
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height

                image_ids.append(img_id)
                classes.append(class_names[int(class_id)])
                boxes.append([x1, y1, x2, y2])

    # Create DataFrame
    train_df = pd.DataFrame(
        {
            "Image_ID": image_ids,
            "class": classes,
            "xmin": [box[0] for box in boxes],
            "ymin": [box[1] for box in boxes],
            "xmax": [box[2] for box in boxes],
            "ymax": [box[3] for box in boxes],
        }
    )

    # Initialize dataset and dataloader
    train_collator = Collator(
        train_augment_and_transform(
            (processor.size["height"], processor.size["width"]),
            config["transforms"]["train"],
        ),
        processor,
    )
    train_dataset = MalariaDataset(train_df, os.path.join(train_path, "images"))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=train_collator,
    )

    return train_dataloader
