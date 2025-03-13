import logging

import pandas as pd #type: ignore
import torch #type:ignore
from torchvision.ops import nms #type:ignore

from postprocessing.postprocess_functions import (
    postprocessing_pipeline,
    ensemble_class_specific_pipeline,
)


def _apply_nms_to_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> tuple:
    """Apply NMS to boxes, handling all classes together."""
    # Apply NMS across all classes at once
    keep = nms(boxes, scores, iou_threshold)

    return keep


def apply_nms_to_df(df, iou_threshold, score_col="confidence"):
    df = df.copy()
    boxes = df[["xmin", "ymin", "xmax", "ymax"]].values
    scores = df[score_col].values
    labels = df["class"].map({"Trophozoite": 0, "WBC": 1}).to_numpy()
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    labels = torch.tensor(labels)
    keep = _apply_nms_to_boxes(boxes, scores, labels, iou_threshold)
    df = df.iloc[keep]
    return df


def run_postprocessing(config, fold_num, yolo11_cv_files_split, detr_cv_files):
    logging.info(f"Running fold {fold_num}")

    def create_pipeline_config(stage_config, base_paths):
        pipeline_config = {
            "DATA_DIR": base_paths["DATA_DIR"],
            "NEG_CSV": base_paths["NEG_CSV"],
            "TEST_CSV": base_paths["TEST_CSV"],
            "TRAIN_CSV": base_paths["TRAIN_CSV"],
            "SPLIT_CSV": base_paths["SPLIT_CSV"],
            "fold_num": fold_num,
        }

        # Handle class-specific parameters
        if "Trophozoite_params" in stage_config:
            for key, value in stage_config["Trophozoite_params"].items():
                pipeline_config[f"Trophozoite_{key}"] = value
        if "WBC_params" in stage_config:
            for key, value in stage_config["WBC_params"].items():
                pipeline_config[f"WBC_{key}"] = value

        # Add any non-class-specific parameters
        for key, value in stage_config.items():
            if key not in ["Trophozoite_params", "WBC_params"]:
                pipeline_config[key] = value

        return pipeline_config

    # Process YOLO predictions
    yolo_dfs = []
    # Process YOLO11
    logging.info(
        f"Processing YOLO11 predictions for fold {fold_num}. Will run TTA ensemble."
    )
    for tta_flip in range(len(yolo11_cv_files_split)):
        yolo_df = pd.read_csv(yolo11_cv_files_split[tta_flip])
        yolo_dfs.append(yolo_df)
    yolo_tta_config = create_pipeline_config(
        config["postprocessing"]["ensemble_ttayolo"], config["input"]
    )

    yolo_tta_df = ensemble_class_specific_pipeline(
        CONFIG=yolo_tta_config,
        df_list=yolo_dfs,
        weight_list=[[1, 1, 1, 1], [1, 1, 1, 1]],
    )  # weight list expects 4 weights for troph and wbc
    logging.info(f"Completed YOLO11 ensembling of TTA predictions for fold {fold_num}")
    yolo_individual_config = create_pipeline_config(
        config["postprocessing"]["individual_yolo11"], config["input"]
    )
    yolo_tta_df = postprocessing_pipeline(yolo_individual_config, yolo_tta_df)
    logging.info(f"Completed YOLO11 postprocessing for fold {fold_num}")

    # Process DETR predictions
    detr_dfs = []
    for tta_flip in range(len(detr_cv_files)):
        detr_df = pd.read_csv(detr_cv_files[tta_flip])
        detr_df_nms = apply_nms_to_df(detr_df, 0.6)
        detr_dfs.append(detr_df_nms)

    detr_tta_config = create_pipeline_config(
        config["postprocessing"]["ensemble_ttadetr"], config["input"]
    )
    detr_tta_df = ensemble_class_specific_pipeline(
        CONFIG=detr_tta_config,
        df_list=detr_dfs,
        weight_list=[[1, 1, 1, 1], [1, 1, 1, 1]],
    )  # weight list expects 4 weights for troph and wbc
    logging.info(f"Completed DETR ensembling of TTA predictions for fold {fold_num}")

    detr_pipeline_config = create_pipeline_config(
        config["postprocessing"]["individual_detr"], config["input"]
    )

    detr_df = postprocessing_pipeline(detr_pipeline_config, detr_tta_df)

    # Final ensemble
    logging.info(f"Running final ensemble for fold {fold_num}")
    final_pipeline_config = create_pipeline_config(
        config["postprocessing"]["ensemble_all"], config["input"]
    )
    final_weights = [[1, 1], [1, 1]]

    all_df = ensemble_class_specific_pipeline(
        CONFIG=final_pipeline_config,
        df_list=[yolo_tta_df, detr_df],
        weight_list=final_weights,
    )
    all_df = postprocessing_pipeline(final_pipeline_config, all_df)
    # Calculate metrics
    return all_df
