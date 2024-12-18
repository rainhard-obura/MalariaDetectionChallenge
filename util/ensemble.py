from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Box:
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float
    confidence: float
    class_name: str
    model_index: Optional[int] = None

    def __eq__(self, other):
        return (
            self.x1 == other.x1
            and self.y1 == other.y1
            and self.x2 == other.x2
            and self.y2 == other.y2
            and self.class_name == other.class_name
        )


@dataclass
class ImagePrediction:
    boxes: list[Box]


Group = list[Box]


def _find_iou(box1: Box, box2: Box) -> float:
    """Calculate the Intersection over Union of two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1)
    box2_area = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1)
    union = box1_area + box2_area - intersection
    return intersection / float(union)


class DualEnsemble:
    def __init__(
        self,
        form: str,
        iou_threshold: float,
        conf_threshold: float,
        classes: list[str],
        conf_threshold2=None,
        **kwargs,
    ):
        assert (
            conf_threshold2 is None or conf_threshold2 >= conf_threshold
        ), "conf_threshold2 must be greater than conf_threshold"
        if conf_threshold2 is None:
            conf_threshold2 = conf_threshold
        self.ensembles = [
            Ensemble(
                form,
                iou_threshold,
                conf_threshold,
                classes=classes,
                threshold_type="lower",
                **kwargs,
            ),
            Ensemble(
                form,
                iou_threshold,
                conf_threshold2,
                classes=classes,
                threshold_type="upper",
                **kwargs,
            ),
        ]

    def __call__(self, preds: list[pd.DataFrame]) -> pd.DataFrame:
        lower_preds = self.ensembles[0](preds)
        upper_preds = self.ensembles[1](preds)
        preds = pd.concat([lower_preds, upper_preds])
        return preds


class Ensemble:
    def __init__(
        self,
        form: str,
        iou_threshold: float,
        conf_threshold: float,
        classes: list[str],
        threshold_type: str,
        **kwargs,
    ):
        self.form = form
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.threshold_type = threshold_type
        self.params = kwargs
        self.n_models = None

    def __call__(self, preds: list[pd.DataFrame]) -> pd.DataFrame:
        self.n_models = len(preds)
        image_predictions = [
            self._to_image_predictions(pred, i) for i, pred in enumerate(preds)
        ]
        groups = self._find_groups(image_predictions)
        for img_id in groups:
            post_reduce_boxes = []
            for group in groups[img_id]:
                post_reduce_boxes.extend(self._reduce_group(group))
            groups[img_id] = post_reduce_boxes
        return self._to_df(groups)

    def _to_df(self, post_reduce_boxes: dict[str, list[Box]]) -> pd.DataFrame:
        rows = []
        for img_id, boxes in post_reduce_boxes.items():
            for box in boxes:
                rows.append(
                    {
                        "Image_ID": img_id,
                        "xmin": box.x1,
                        "ymin": box.y1,
                        "xmax": box.x2,
                        "ymax": box.y2,
                        "confidence": box.confidence,
                        "class": box.class_name,
                    }
                )
        return pd.DataFrame(rows)

    def _to_image_predictions(
        self, pred: pd.DataFrame, model_index: int
    ) -> dict[str, ImagePrediction]:
        image_predictions = {}
        for img_id in pred["Image_ID"].unique():
            boxes_df = pred[pred["Image_ID"] == img_id]
            boxes = []
            for _, box in boxes_df.iterrows():
                if (
                    self.threshold_type == "lower"
                    and box.confidence < self.conf_threshold
                ):
                    continue
                elif (
                    self.threshold_type == "upper"
                    and box.confidence > self.conf_threshold
                ):
                    continue
                elif box.confidence < 0.05:
                    continue
                boxes.append(
                    Box(
                        box.xmin,
                        box.ymin,
                        box.xmax,
                        box.ymax,
                        box.confidence,
                        box["class"],
                        model_index,
                    )
                )
            image_predictions[img_id] = ImagePrediction(boxes)
        return image_predictions

    def _find_groups_in_image(
        self, image_predictions: list[ImagePrediction]
    ) -> list[Group]:
        groups = []
        for class_name in self.classes:
            all_boxes = [
                box
                for pred in image_predictions
                if pred is not None
                for box in pred.boxes
                if box.class_name == class_name
            ]
            all_boxes = sorted(all_boxes, key=lambda box: box.x1)
            while len(all_boxes) > 0:
                box_indices = {0}
                checked_indices = set()
                while box_indices != checked_indices:
                    to_check = box_indices - checked_indices
                    checked_indices = box_indices.copy()
                    for index in to_check:
                        for other_index in range(len(all_boxes)):
                            # once a box is completely to the right of the current box, break
                            if all_boxes[other_index].x1 > all_boxes[index].x2:
                                break
                            if (
                                _find_iou(all_boxes[index], all_boxes[other_index])
                                > self.iou_threshold
                            ):
                                box_indices.add(other_index)
                groups.append([all_boxes[i] for i in box_indices])
                all_boxes = [
                    box for i, box in enumerate(all_boxes) if i not in box_indices
                ]
        return groups

    def _find_groups(
        self, preds: list[dict[str, ImagePrediction]]
    ) -> dict[str, list[Group]]:
        # Modified to handle missing predictions for certain images
        image_ids = set().union(
            *(pred.keys() for pred in preds)
        )  # Get all unique image IDs across predictions
        groups = {img_id: [] for img_id in image_ids}
        for img_id in image_ids:
            valid_predictions = [
                pred[img_id] for pred in preds if img_id in pred
            ]  # Only use predictions that contain the current img_id
            if valid_predictions:
                groups[img_id] = self._find_groups_in_image(valid_predictions)
        return groups

    def _reduce_group(self, group: Group) -> Group:
        if self.form == "nms":
            return [self._nms(group)]
        elif self.form == "wbf":
            return [self._weighted_boxes_fusion(group)]
        elif self.form == "soft_nms":
            return self._soft_nms(group)
        elif self.form == "voting":
            return self._voting(group)
        else:
            raise ValueError(f"Unknown form: {self.form}")

    def _nms(self, group: Group) -> Box:
        return max(group, key=lambda box: box.confidence)

    def _soft_nms(self, group: Group) -> Group:
        base_box = self._nms(group)
        for box in group:
            if box == base_box:
                continue
            iou = _find_iou(box, base_box)
            box.confidence *= 1 - iou
        return group

    def _voting(self, group: Group) -> Group:
        if len(set([box.model_index for box in group])) > 0.5 * self.n_models:
            return [self._nms(group)]
        else:
            return []

    def _weighted_boxes_fusion(self, group: Group) -> Box:
        x1 = np.array([box.x1 for box in group])
        y1 = np.array([box.y1 for box in group])
        x2 = np.array([box.x2 for box in group])
        y2 = np.array([box.y2 for box in group])
        confidences = np.array([box.confidence for box in group])

        # Calculate the weighted average of the box coordinates
        x1_avg = np.average(x1, weights=confidences)
        y1_avg = np.average(y1, weights=confidences)
        x2_avg = np.average(x2, weights=confidences)
        y2_avg = np.average(y2, weights=confidences)

        # calculate confidence score
        n_models = len(self.params["weights"])
        model_ids = [box.model_index for box in group]
        reduction_method = self.params.get("wbf_reduction")
        if reduction_method == "mean":
            conf = 0
            for confidence, model_id in zip(confidences, model_ids):
                conf += (
                    confidence
                    * self.params["weights"][model_id]
                    / model_ids.count(model_id)
                )
            conf /= n_models
        elif reduction_method == "max":
            conf = max(confidences)
        elif reduction_method == "min":
            conf = min(confidences)
        elif reduction_method == "median":
            conf = np.median(confidences)
        elif reduction_method == "mean_no_rescale":
            conf = np.average(confidences)
        elif reduction_method == "random":
            conf = np.random.choice(confidences)
        else:
            raise ValueError(f"Unknown reduction method: {reduction_method}")

        # Create the fused bounding box
        fused_box = Box(
            x1=float(x1_avg),
            y1=float(y1_avg),
            x2=float(x2_avg),
            y2=float(y2_avg),
            confidence=float(conf),
            class_name=group[0].class_name,
        )

        return fused_box
