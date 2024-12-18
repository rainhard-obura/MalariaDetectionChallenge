from typing import List

import cv2
import pandas as pd
import tqdm

from inference.tta import get_img_tta_augs


def yolo_predict_tta(model, img_paths) -> List[pd.DataFrame]:
    tta_preds = [[] for _ in range(4)]

    for image_dir in tqdm.tqdm(img_paths, desc="predicting with YOLO"):
        image_id = image_dir.split("/")[-1]
        image = cv2.imread(image_dir)
        h, w = image.shape[:2]

        for i, im_aug in enumerate(get_img_tta_augs(image)):
            predictions = model(
                im_aug, conf=0.0, device=0, verbose=False, augment=False
            )

            final_predictions = []
            for r in predictions:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    class_name = model.names[cls]

                    xmin, ymin, xmax, ymax = xyxy
                    if i == 1 or i == 3:
                        ymin, ymax = h - ymax, h - ymin

                    if i == 2 or i == 3:
                        xmin, xmax = w - xmax, w - xmin

                    final_predictions.append(
                        {
                            "Image_ID": image_id,
                            "class": class_name,
                            "confidence": conf,
                            "ymin": ymin,
                            "xmin": xmin,
                            "ymax": ymax,
                            "xmax": xmax,
                        }
                    )

                if len(boxes) == 0:
                    final_predictions.append(
                        {
                            "Image_ID": image_id,
                            "class": "NEG",
                            "confidence": 0,
                            "ymin": 0,
                            "xmin": 0,
                            "ymax": 0,
                            "xmax": 0,
                        }
                    )

            tta_preds[i] += final_predictions

    final_tta_preds = [pd.DataFrame(aug_preds) for aug_preds in tta_preds]
    return final_tta_preds
