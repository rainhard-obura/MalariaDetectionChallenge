import numpy as np
import pandas as pd
from tqdm import tqdm


def _bb_intersection_over_union(boxA, boxB):
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


def _non_max_suppression(boxes, scores, threshold):
    # Sort by confidence score
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        ious = np.array(
            [_bb_intersection_over_union(boxes[i], boxes[j]) for j in order[1:]]
        )

        inds = np.where(ious <= threshold)[0]
        order = order[inds + 1]

    return keep


def apply_nms(df, iou_threshold=0.5):
    # Separate positive and negative detections
    df_pos = df[df["class"] != "NEG"]
    df_neg = df[df["class"] == "NEG"]

    print(df_neg.shape)

    # Group positive detections by image
    grouped = df_pos.groupby("Image_ID")

    results = []
    for image_id, group in tqdm(grouped):
        boxes = group[["xmin", "ymin", "xmax", "ymax"]].values.astype(float)
        scores = group["confidence"].values.astype(float)

        keep = _non_max_suppression(boxes, scores, iou_threshold)

        # Keep the selected detections
        results.append(group.iloc[keep])

    # Combine results
    df_result = pd.concat(results + [df_neg], ignore_index=True)

    print(f"Number of boxes before NMS: {len(df)}")
    print(f"Number of boxes after NMS: {len(df_result)}")

    return df_result
