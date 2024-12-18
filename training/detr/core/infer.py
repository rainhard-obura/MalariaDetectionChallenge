import os
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .constants import ID2LABEL


@dataclass
class Prediction:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]


@dataclass
class ImagePrediction:
    img_name: str
    predictions: List[Prediction]


def infer(
    cfg,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    processor: torch.nn.Module = None,
) -> tuple:
    """
    Perform inference on the given data loader using the provided model.

    Args:
        cfg: The configuration object.
        model: The model to use for inference.
        data_loader: DataLoader containing the inference data.
        device: The device to use for inference.
        processor: The image processor.

    Returns:
        A tuple containing the prediction DataFrame and the average loss.
    """
    model.eval()
    results = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inferring"):
            img_names = batch["img_names"]
            images = batch["pixel_values"].to(device)
            labels = (
                [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                if "labels" in batch
                else None
            )
            target_sizes = torch.tensor(
                [
                    Image.open(os.path.join(cfg.data.image_dir, img_name)).size[::-1]
                    for img_name in img_names
                ]
            )

            outputs = model(images, labels=labels)

            if labels is not None:
                total_loss += outputs.loss.item()
                num_batches += 1

            processed_results = processor.post_process_object_detection(
                outputs,
                threshold=cfg.testing.confidence_threshold,
                target_sizes=target_sizes,
            )

            for img_name, result in zip(img_names, processed_results):
                for score, label, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    results.append(
                        {
                            "Image_ID": img_name,
                            "class": ID2LABEL[label.item()],
                            "confidence": score.item(),
                            "xmin": box[0].item(),
                            "ymin": box[1].item(),
                            "xmax": box[2].item(),
                            "ymax": box[3].item(),
                        }
                    )

    predictions_df = pd.DataFrame(results)
    avg_loss = total_loss / num_batches if num_batches > 0 else None

    return predictions_df, avg_loss
