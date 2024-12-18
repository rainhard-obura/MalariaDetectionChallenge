import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from ..constants import LABEL2ID


@dataclass
class DatasetItem:
    img_name: str
    image: Image.Image
    target: Dict[str, List]


class MalariaDataset(Dataset):
    def __init__(self, df, image_dir, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.images = df["Image_ID"].unique().tolist()
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.images)

    def get_image_path(self, image_id: str) -> str:
        return os.path.join(self.image_dir, f"{image_id}")

    def __getitem__(self, idx: int) -> DatasetItem:
        img_name = self.images[idx]
        img_path = self.get_image_path(img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get annotations for the image
        if self.is_test:
            target = {"boxes": [], "labels": [], "area": []}
        else:
            boxes = self.df[self.df["Image_ID"] == img_name]
            target = self._prepare_target(boxes)

        return DatasetItem(img_name=img_name, image=image, target=target)

    def _prepare_target(self, boxes: pd.DataFrame) -> Dict[str, List]:
        target = {"boxes": [], "labels": [], "area": []}
        for _, box in boxes.iterrows():
            if box["class"] != "NEG":
                bbox = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
                target["boxes"].append(bbox)
                target["labels"].append(LABEL2ID[box["class"]])
                target["area"].append(
                    (box["xmax"] - box["xmin"]) * (box["ymax"] - box["ymin"])
                )
        return target


class Collator:
    def __init__(self, transforms, processor):
        self.transforms = transforms
        self.processor = processor

    def __call__(self, batch: List[DatasetItem]) -> Dict[str, torch.Tensor]:
        img_names, images, targets = zip(
            *[(item.img_name, item.image, item.target) for item in batch]
        )

        # Resize images and adjust bounding boxes
        images, targets = self._resize_images_and_targets(images, targets)

        # Convert images to tensors
        images = [T.ToTensor()(img) for img in images]  # Shape: (C, H, W)
        images = torch.stack(images)  # Shape: (B, C, H, W)

        if not targets[0]:  # Check if targets are empty (test set)
            # Prepare dummy annotations for the processor
            batch_annotations = [
                {"image_id": idx, "annotations": []} for idx in range(len(batch))
            ]
        else:
            # Prepare bounding boxes and labels
            bboxes_list = [
                (
                    target["boxes"].clone().detach().to(dtype=torch.float32)
                    if isinstance(target["boxes"], torch.Tensor)
                    else torch.tensor(target["boxes"], dtype=torch.float32)
                )
                for target in targets
            ]
            labels_list = [
                (
                    target["labels"].clone().detach().to(dtype=torch.int64)
                    if isinstance(target["labels"], torch.Tensor)
                    else torch.tensor(target["labels"], dtype=torch.int64)
                )
                for target in targets
            ]

            # Convert bounding boxes to the required format
            bboxes_list = [self._convert_bbox_format(bbox) for bbox in bboxes_list]

            # Apply augmentations
            if self.transforms:
                images, bboxes_list = self.transforms(images, bboxes_list)

            # Prepare targets in COCO format for the processor
            batch_annotations = []
            for idx, (bboxes, labels) in enumerate(zip(bboxes_list, labels_list)):
                if bboxes.numel() > 0:
                    areas = (bboxes[:, 2, 0] - bboxes[:, 0, 0]) * (
                        bboxes[:, 2, 1] - bboxes[:, 0, 1]
                    )
                    bboxes_xyxy = self._convert_bbox_format_back(bboxes).tolist()
                else:
                    areas = []
                    bboxes_xyxy = []

                batch_annotations.append(
                    self._format_image_annotations_as_coco(
                        image_id=idx,
                        categories=labels.tolist(),
                        areas=areas,
                        bboxes=bboxes_xyxy,
                    )
                )

        # Use the processor
        encoding = self.processor(
            images=list(images), annotations=batch_annotations, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"]
        target = encoding["labels"] if "labels" in encoding else None

        return {"pixel_values": pixel_values, "labels": target, "img_names": img_names}

    def _convert_bbox_format(self, bbox):
        """Convert bounding box from (N, 4) to (N, 4, 2) format."""
        if bbox.numel() == 0:  # No detections
            return torch.zeros((0, 4, 2), dtype=bbox.dtype, device=bbox.device)

        N, _ = bbox.shape
        new_bbox = torch.zeros(N, 4, 2)
        new_bbox[:, 0, 0] = bbox[:, 0]  # x_min
        new_bbox[:, 0, 1] = bbox[:, 1]  # y_min
        new_bbox[:, 1, 0] = bbox[:, 2]  # x_max
        new_bbox[:, 1, 1] = bbox[:, 1]  # y_min
        new_bbox[:, 2, 0] = bbox[:, 2]  # x_max
        new_bbox[:, 2, 1] = bbox[:, 3]  # y_max
        new_bbox[:, 3, 0] = bbox[:, 0]  # x_min
        new_bbox[:, 3, 1] = bbox[:, 3]  # y_max
        return new_bbox

    def _convert_bbox_format_back(self, bbox):
        """Convert bounding box from (B, N, 4, 2) back to (B, N, 4) format."""
        if bbox.numel() == 0:  # No detections
            return torch.zeros((0, 4), dtype=bbox.dtype, device=bbox.device)

        N, _, _ = bbox.shape
        new_bbox = torch.zeros(N, 4)
        new_bbox[:, 0] = bbox[:, 0, 0]  # x_min
        new_bbox[:, 1] = bbox[:, 0, 1]  # y_min
        new_bbox[:, 2] = bbox[:, 2, 0]  # x_max
        new_bbox[:, 3] = bbox[:, 2, 1]  # y_max

        return new_bbox

    def _resize_images_and_targets(self, images, targets):
        """
        Resize images and corresponding targets to the specified dimensions.

        Args:
            images (List[PIL.Image]): List of input images.
            targets (List[Dict]): List of target dictionaries containing bounding boxes and labels.

        Returns:
            Tuple[List[PIL.Image], List[Dict]]: Resized images and updated targets.
        """
        resized_images = []
        updated_targets = []

        for image, target in zip(images, targets):
            # Resize image
            resized_image = image.resize(
                (self.processor.size["width"], self.processor.size["height"])
            )

            # Calculate scaling factors
            w_scale = self.processor.size["width"] / image.width
            h_scale = self.processor.size["height"] / image.height

            # Update bounding boxes
            updated_boxes = []
            for box in target["boxes"]:
                x_min, y_min, x_max, y_max = box
                updated_box = [
                    x_min * w_scale,
                    y_min * h_scale,
                    x_max * w_scale,
                    y_max * h_scale,
                ]
                updated_boxes.append(updated_box)

            # Update target
            updated_target = target.copy()
            updated_target["boxes"] = torch.tensor(updated_boxes)

            resized_images.append(resized_image)
            updated_targets.append(updated_target)

        return resized_images, updated_targets

    def _format_image_annotations_as_coco(
        self,
        image_id: int,
        categories: List[int],
        areas: List[float],
        bboxes: List[List[float]],
    ) -> Dict:
        """
        Format one set of image annotations to the COCO format.
        """
        annotations = [
            {
                "image_id": image_id,
                "category_id": category,
                "iscrowd": 0,
                "area": area,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            }
            for category, area, (x_min, y_min, x_max, y_max) in zip(
                categories, areas, bboxes
            )
        ]

        return {
            "image_id": image_id,
            "annotations": annotations,
        }
