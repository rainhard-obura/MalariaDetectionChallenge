import os

import pandas as pd
import torch
import tqdm
import yaml
from PIL import Image
from einops import rearrange
from torch.utils.data import DataLoader

from inference.tta import get_img_tta_augs, deaugment_boxes
from .core.datasets.detr_dataset import MalariaDataset, Collator
from .core.model import get_processor


def detr_predict_tta(model, config_file, img_paths, conf: float = 0.0):
    # Setup processor
    config_dict = dict(yaml.safe_load(open(config_file)))
    device = config_dict["device"]
    processor = get_processor(config_dict.pop("processor"))
    batch_size = config_dict["testing"]["test_batch_size"]

    # Create a DataFrame from image paths
    df = pd.DataFrame({"Image_ID": [os.path.basename(path) for path in img_paths]})

    img_dir = os.path.dirname(img_paths[0])  # Assume all images are in same directory

    # Create dataset and dataloader
    dataset = MalariaDataset(df=df, image_dir=img_dir, is_test=True)

    collator = Collator(transforms=None, processor=processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    tta_preds = [[] for _ in range(4)]
    model.eval()

    # Process images in batches
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="predicting with DETR"):
            batch_images = batch["pixel_values"]  # [B, C, H, W]
            batch_ids = batch["img_names"]

            batch_images_np = [
                rearrange(img, "c h w -> h w c")
                .cpu()
                .numpy()  # Just convert to numpy in HWC format
                for img in batch_images
            ]

            # Process each TTA variant
            for aug_idx in range(4):
                # Apply TTA to all images in batch
                batch_aug_images = [
                    get_img_tta_augs(img)[
                        aug_idx
                    ]  # get_img_tta_augs works with normalized numpy arrays
                    for img in batch_images_np
                ]

                # Convert back to tensor format [B, C, H, W]
                batch_aug_images = [
                    rearrange(
                        torch.from_numpy(img), "h w c -> c h w"
                    )  # Convert back to CHW format
                    for img in batch_aug_images
                ]
                batch_aug_images = torch.stack(batch_aug_images).to(device)

                outputs = model(pixel_values=batch_aug_images)

                # Get original image sizes for predictions
                orig_sizes = torch.tensor(
                    [
                        Image.open(os.path.join(img_dir, img_id)).size[::-1]
                        for img_id in batch_ids
                    ]
                ).to(device)

                # Process each image's results in the batch
                processed_results = processor.post_process_object_detection(
                    outputs, threshold=conf, target_sizes=orig_sizes
                )

                # Process predictions for each image in batch
                for img_idx, (image_id, results) in enumerate(
                    zip(batch_ids, processed_results)
                ):
                    final_predictions = []

                    if len(results["boxes"]) > 0:
                        # Get image dimensions for deaugmentation
                        img = Image.open(os.path.join(img_dir, image_id))
                        h, w = img.size[::-1]  # height, width

                        # Convert boxes to numpy for easier manipulation
                        boxes = results["boxes"].cpu().numpy()

                        # Deaugment the boxes
                        boxes = deaugment_boxes(boxes, h, w, aug_idx)

                        # Create predictions with deaugmented boxes
                        for score, label, box in zip(
                            results["scores"], results["labels"], boxes
                        ):
                            pred_dict = {
                                "Image_ID": image_id,
                                "class": model.config.id2label[label.item()],
                                "confidence": score.item(),
                                "xmin": box[0],
                                "ymin": box[1],
                                "xmax": box[2],
                                "ymax": box[3],
                            }
                            final_predictions.append(pred_dict)

                    if len(results["boxes"]) == 0:
                        neg_pred = {
                            "Image_ID": image_id,
                            "class": "NEG",
                            "confidence": 0,
                            "ymin": 0,
                            "xmin": 0,
                            "ymax": 0,
                            "xmax": 0,
                        }
                        final_predictions.append(neg_pred)

                    tta_preds[aug_idx] += [list(f.values()) for f in final_predictions]

    tta_preds = [
        pd.DataFrame(
            aug_preds,
            columns=["Image_ID", "class", "confidence", "xmin", "ymin", "xmax", "ymax"],
        )
        for aug_preds in tta_preds
    ]

    return tta_preds
