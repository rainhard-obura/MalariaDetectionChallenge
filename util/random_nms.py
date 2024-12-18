import random

import numpy as np
import pandas as pd
from tqdm import tqdm


def bb_intersection_over_union(boxA, boxB):
    """
    Calculate IoU between two boxes

    Args:
        boxA: numpy array or list [xmin, ymin, xmax, ymax]
        boxB: numpy array or list [xmin, ymin, xmax, ymax]
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def random_nms(df, threshold, seed=None):
    """
    Performs class-specific non-maximum suppression with random selection on a DataFrame of boxes.
    Only removes overlapping boxes of the same class.

    Args:
        df: DataFrame with columns ['Image_ID', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
        threshold: IoU threshold for suppressing overlapping boxes
        seed: Random seed for reproducibility

    Returns:
        DataFrame with NMS applied
    """
    if seed is not None:
        random.seed(seed)

    # Process each image separately
    image_ids = df["Image_ID"].unique()
    results = []

    for img_id in tqdm(image_ids, desc="Processing images"):
        # Get boxes for current image
        img_df = df[df["Image_ID"] == img_id].copy()

        if len(img_df) <= 1:
            results.append(img_df)
            continue

        # Get unique classes in this image
        classes = img_df["class"].unique()

        # Process each class separately
        keep_indices = []

        for cls in classes:
            # Get indices for this class
            class_mask = img_df["class"] == cls
            class_indices = class_mask[class_mask].index.tolist()

            if len(class_indices) <= 1:
                keep_indices.extend(class_indices)
                continue

            # Get boxes for this class
            class_boxes = []
            for idx in class_indices:
                row = img_df.loc[idx]
                class_boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            class_boxes = np.array(class_boxes)

            # Random NMS for this class
            indices = list(range(len(class_boxes)))
            random.shuffle(indices)  # Randomize the order
            class_keep = []

            while indices:
                i = indices.pop(0)  # Take the first box from shuffled list
                class_keep.append(i)

                remaining_indices = []
                for j in indices:
                    iou = bb_intersection_over_union(class_boxes[i], class_boxes[j])
                    if iou <= threshold:
                        remaining_indices.append(j)

                indices = remaining_indices

            # Convert class-specific indices back to DataFrame indices
            keep_indices.extend([class_indices[i] for i in class_keep])

        # Keep the selected boxes
        results.append(img_df.loc[keep_indices])

    # Combine all results
    filtered_df = pd.concat(results, ignore_index=True)

    print(f"Original number of boxes: {len(df)}")
    print(f"Number of boxes after class-specific NMS: {len(filtered_df)}")

    return filtered_df


def get_filtered_train_df(df, iou_threshold=0.5, seed=None):
    """
    Apply class-specific random NMS to filter training data

    Args:
        df: DataFrame with annotations
        iou_threshold: IoU threshold for NMS
        seed: Random seed for reproducibility

    Returns:
        Filtered DataFrame
    """
    # Separate positive and negative detections
    df_pos = df[df["class"] != "NEG"]
    df_neg = df[df["class"] == "NEG"]

    # Apply random NMS to positive detections
    df_filtered = random_nms(df_pos, iou_threshold, seed)

    # Combine with negative detections
    result = pd.concat([df_filtered, df_neg], ignore_index=True)

    return result
