from typing import List

import cv2
import numpy as np


def get_img_tta_augs(image: cv2.Mat) -> List[cv2.Mat]:
    tta_images = [
        image,  # original
        cv2.flip(image, 0),  # vertical flip
        cv2.flip(image, 1),  # horizontal flip
        cv2.flip(cv2.flip(image, 0), 1),  # both flips
    ]
    return tta_images


def deaugment_boxes(
    boxes: np.ndarray, image_height: int, image_width: int, aug_idx: int
) -> np.ndarray:
    """
    Reverse the augmentation effect on bounding boxes
    boxes: numpy array of shape (N, 4) with [xmin, ymin, xmax, ymax]
    aug_idx: 0=original, 1=vertical flip, 2=horizontal flip, 3=both flips
    """
    boxes = boxes.copy()
    if aug_idx == 0:  # original
        return boxes
    elif aug_idx == 1:  # vertical flip
        boxes[:, [1, 3]] = image_height - boxes[:, [3, 1]]
    elif aug_idx == 2:  # horizontal flip
        boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]]
    elif aug_idx == 3:  # both flips
        boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]]
        boxes[:, [1, 3]] = image_height - boxes[:, [3, 1]]
    return boxes
