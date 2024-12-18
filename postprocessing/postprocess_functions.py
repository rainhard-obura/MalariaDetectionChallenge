from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from util.ensemble import DualEnsemble

# global cache variable
cached_kde_wbc = None
cached_kde_troph = None


def get_img_shape(data_dir, image_id):
    return cv2.imread(f"{data_dir}/{image_id}").shape


def add_img_shape_column(data_dir, df):
    unique_image_ids = df["Image_ID"].unique()
    with ThreadPoolExecutor() as executor:
        img_shapes = list(
            executor.map(partial(get_img_shape, data_dir), unique_image_ids)
        )
    shape_dict = dict(zip(unique_image_ids, img_shapes))
    df["img_shape"] = df["Image_ID"].map(shape_dict)
    return df


def no_wbc_labels_in_certain_size(df):
    certain_size = (3120, 4160, 3)
    mask = (
        df["img_shape"].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)
        == certain_size
    ) & (df["class"] == "WBC")
    return df[~mask]


def assign_neg_class(neg_csv, df):
    neg_df = pd.read_csv(neg_csv)
    neg_ids = neg_df[neg_df["class"] == "NEG"]["Image_ID"].unique()
    df.loc[df["Image_ID"].isin(neg_ids), "class"] = "NEG"
    return df


def add_negs_to_submission(df, neg_csv, test_csv):
    neg_df = pd.read_csv(neg_csv)
    test_df = pd.read_csv(test_csv)
    neg_ids = neg_df.loc[neg_df["class"] == "NEG", "Image_ID"].unique()
    test_ids = test_df["Image_ID"].unique()
    neg_ids = np.intersect1d(neg_ids, test_ids)
    df = df[~df["Image_ID"].isin(neg_ids)]
    neg_rows = []
    for image_id in neg_ids:
        neg_row = {
            "Image_ID": image_id,
            "class": "NEG",
            "confidence": 1,
            "ymin": 0,
            "xmin": 0,
            "ymax": 0,
            "xmax": 0,
        }
        neg_rows.append(neg_row)
    neg_rows_df = pd.DataFrame(neg_rows)
    df = pd.concat([df, neg_rows_df], ignore_index=True)
    return df


def process_df_bbox(df, img_folder):
    boxes_df = add_img_shape_column(img_folder, df)
    boxes_df["img_width"] = boxes_df["img_shape"].apply(lambda x: x[1])
    boxes_df["img_height"] = boxes_df["img_shape"].apply(lambda x: x[0])
    boxes_df["x_center"] = (boxes_df["xmin"] + boxes_df["xmax"]) / 2
    boxes_df["y_center"] = (boxes_df["ymin"] + boxes_df["ymax"]) / 2
    boxes_df["width"] = boxes_df["xmax"] - boxes_df["xmin"]
    boxes_df["height"] = boxes_df["ymax"] - boxes_df["ymin"]
    boxes_df["area"] = boxes_df["width"] * boxes_df["height"]
    boxes_df["aspect_ratio"] = boxes_df["width"] / boxes_df["height"]
    boxes_df["norm_center_x"] = boxes_df["x_center"] / boxes_df["img_width"]
    boxes_df["norm_center_y"] = boxes_df["y_center"] / boxes_df["img_height"]
    boxes_df["norm_width"] = boxes_df["width"] / boxes_df["img_width"]
    boxes_df["norm_height"] = boxes_df["height"] / boxes_df["img_height"]
    boxes_df["norm_area"] = boxes_df["norm_width"] * boxes_df["norm_height"]
    return boxes_df


def update_bbox_from_yolo(df):
    df = df.copy()
    df["xmin"] = (df["norm_center_x"] - df["norm_width"] / 2) * df["img_width"]
    df["xmax"] = (df["norm_center_x"] + df["norm_width"] / 2) * df["img_width"]
    df["ymin"] = (df["norm_center_y"] - df["norm_height"] / 2) * df["img_height"]
    df["ymax"] = (df["norm_center_y"] + df["norm_height"] / 2) * df["img_height"]
    df["x_center"] = df["norm_center_x"] * df["img_width"]
    df["y_center"] = df["norm_center_y"] * df["img_height"]
    df["width"] = df["norm_width"] * df["img_width"]
    df["height"] = df["norm_height"] * df["img_height"]
    df["area"] = df["width"] * df["height"]
    df["aspect_ratio"] = df["width"] / df["height"]
    return df


def factor_width_change(df, factor):
    df = df.copy()
    df["norm_width"] = df["norm_width"] * factor
    df = update_bbox_from_yolo(df)
    return df


def factor_height_change(df, factor):
    df = df.copy()
    df["norm_height"] = df["norm_height"] * factor
    df = update_bbox_from_yolo(df)
    return df


def factor_bbox_size_change(df, factor_wbc, factor_troph):
    df = df.copy()
    df_wbc = df[df["class"] == "WBC"]
    df_troph = df[df["class"] == "Trophozoite"]
    df_neg = df[df["class"] == "NEG"]
    df_wbc = factor_width_change(df_wbc, factor_wbc)
    df_wbc = factor_height_change(df_wbc, factor_wbc)
    df_troph = factor_width_change(df_troph, factor_troph)
    df_troph = factor_height_change(df_troph, factor_troph)
    df = pd.concat([df_wbc, df_troph, df_neg], ignore_index=True)
    return df


def remove_bbox_near_edges(df, data_dir, edge_threshold, border_threshold):
    df = df.copy()

    def detect_black_borders(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0, 0, 0, 0
        height, width = img.shape
        top_border = 0
        for y in range(height):
            if np.mean(img[y, :]) > border_threshold:
                break
            top_border = y
        bottom_border = 0
        for y in range(height - 1, -1, -1):
            if np.mean(img[y, :]) > border_threshold:
                break
            bottom_border = height - y
        left_border = 0
        for x in range(width):
            if np.mean(img[:, x]) > border_threshold:
                break
            left_border = x
        right_border = 0
        for x in range(width - 1, -1, -1):
            if np.mean(img[:, x]) > border_threshold:
                break
            right_border = width - x
        return top_border, bottom_border, left_border, right_border

    border_cache = {}
    for img_id in df["Image_ID"].unique():
        img_path = f"{data_dir}/{img_id}"
        border_cache[img_id] = detect_black_borders(img_path)

    def check_bbox_on_boundary(row):
        img_height = row["img_height"]
        img_width = row["img_width"]
        top_border, bottom_border, left_border, right_border = border_cache[
            row["Image_ID"]
        ]
        effective_height = img_height - (top_border + bottom_border)
        effective_width = img_width - (left_border + right_border)
        dist_from_top = (row["ymin"] - top_border) / effective_height
        dist_from_bottom = (img_height - bottom_border - row["ymax"]) / effective_height
        dist_from_left = (row["xmin"] - left_border) / effective_width
        dist_from_right = (img_width - right_border - row["xmax"]) / effective_width
        return not (
            dist_from_top < edge_threshold
            or dist_from_bottom < edge_threshold
            or dist_from_left < edge_threshold
            or dist_from_right < edge_threshold
        )

    df = df[df.apply(check_bbox_on_boundary, axis=1)]
    return df


def apply_confidence_threshold(df, threshold):
    return df[df["confidence"] >= threshold]


def delete_sharp_aspect_ratio(
    df, upper_aspect_ratio_threshold=2, lower_aspect_ratio_threshold=0.5
):
    """
    Delete bounding boxes with sharp aspect ratios while ensuring at least one prediction per image
    """
    df = df.copy()

    # Calculate aspect ratio
    aspect_ratio = (df["xmax"] - df["xmin"]) / (df["ymax"] - df["ymin"])

    # Create mask for valid aspect ratios
    mask = (aspect_ratio < upper_aspect_ratio_threshold) & (
        aspect_ratio > lower_aspect_ratio_threshold
    )

    # Group by image and check if we would remove all predictions
    would_remove_all = df.groupby("Image_ID")["Image_ID"].transform(
        lambda x: (~mask[x.index]).all()
    )

    # Keep predictions even if they violate aspect ratio if they're the only ones for that image
    final_mask = mask | would_remove_all

    return df[final_mask]


def basic_postprocess(df, data_dir, neg_csv, test_csv):
    # submission_df = add_img_shape_column(data_dir, df)
    neg_rows = df[df["class"] == "NEG"]
    filtered_dfs = df[df["class"] != "NEG"]
    filtered_dfs = no_wbc_labels_in_certain_size(filtered_dfs)
    filtered_dfs = filtered_dfs.drop(columns=["img_shape"])
    neg_rows = neg_rows.drop(columns=["img_shape"])
    filtered_dfs = pd.concat([filtered_dfs, neg_rows], ignore_index=True)
    filtered_dfs = add_negs_to_submission(filtered_dfs, neg_csv, test_csv)
    filtered_dfs = assign_neg_class(neg_csv, filtered_dfs)

    return filtered_dfs


def spatial_density_contour_troph(
    df,
    df_train,
    CONFIG,
    option,
    base_adjustment,
    percentile_low,
    percentile_high,
    low_density_adjustment,
    high_density_adjustment,
    expit_scale,
    adjustment_range_low,
    adjustment_range_high,
):
    if "SPLIT_CSV" in CONFIG:
        split_df = pd.read_csv(CONFIG["SPLIT_CSV"])
        df_train = df_train.merge(split_df, on="Image_ID", how="left")
        initialize_troph_kde(df_train[df_train["Split"] == CONFIG["fold_num"]])
    else:
        initialize_troph_kde(df_train)

    # Filter Trophozoite data only
    df_troph = df[df["class"] == "Trophozoite"].copy()
    kde_density_troph = cached_kde_troph.evaluate(
        df_troph[["norm_center_x", "norm_center_y"]].T
    )
    kde_density_norm = kde_density_troph / kde_density_troph.max()

    if option == 0:
        density_multiplier = 1 - base_adjustment
        adjustment = base_adjustment + (density_multiplier * kde_density_norm)
    elif option == 1:
        density_percentile = np.percentile(
            kde_density_norm, [percentile_low, percentile_high]
        )
        adjustment = np.ones_like(kde_density_norm)
        adjustment[kde_density_norm < density_percentile[0]] = low_density_adjustment
        adjustment[kde_density_norm > density_percentile[1]] = high_density_adjustment
    elif option == 2:
        log_scale = np.log1p(kde_density_norm)
        log_scale /= log_scale.max()
        log_scale_factor = 1 - adjustment_range_low
        adjustment = adjustment_range_low + (log_scale_factor * log_scale)
    elif option == 3:
        high_density_threshold = np.percentile(kde_density_norm, percentile_high)
        low_density_threshold = np.percentile(kde_density_norm, percentile_low)
        high_density_mask = kde_density_norm > high_density_threshold
        low_density_mask = kde_density_norm < low_density_threshold
        adjustment = np.ones_like(kde_density_norm)
        adjustment[high_density_mask] = high_density_adjustment
        adjustment[low_density_mask] = low_density_adjustment
        adjustment[np.logical_and(~high_density_mask, ~low_density_mask)] = 0.9
    elif option == 4:
        from scipy.special import expit

        centered_density = kde_density_norm - kde_density_norm.mean()
        smooth_step = expit(centered_density * expit_scale)
        adjustment = adjustment_range_low + ((1 - adjustment_range_low) * smooth_step)

    # Apply adjustment and clip
    df_troph["confidence"] = np.clip(df_troph["confidence"] * adjustment, 0, 1)
    df.loc[df["class"] == "Trophozoite", "confidence"] = df_troph["confidence"].values

    return df


def initialize_WBC_kde(df_train):
    global cached_kde_wbc
    df_train_wbc = df_train[df_train["class"] == "WBC"]
    cached_kde_wbc = gaussian_kde(df_train_wbc[["norm_center_x", "norm_center_y"]].T)


def initialize_troph_kde(df_train):
    global cached_kde_troph
    df_train_troph = df_train[df_train["class"] == "Trophozoite"]
    cached_kde_troph = gaussian_kde(
        df_train_troph[["norm_center_x", "norm_center_y"]].T
    )


def spatial_density_contour_wbc(
    df,
    df_train,
    CONFIG,
    option,
    base_adjustment,
    density_multiplier,
    percentile_low,
    percentile_high,
    low_density_adjustment,
    high_density_adjustment,
    log_scale_factor,
    expit_scale,
    adjustment_range_low,
    adjustment_range_high,
):
    if "SPLIT_CSV" in CONFIG:
        split_df = pd.read_csv(CONFIG["SPLIT_CSV"])
        df_train = df_train.merge(split_df, on="Image_ID", how="left")
        initialize_WBC_kde(df_train[df_train["Split"] == CONFIG["fold_num"]])
    else:
        initialize_WBC_kde(df_train)
    # Filter WBC data only
    df_wbc = df[df["class"] == "WBC"].copy()
    kde_density_wbc = cached_kde_wbc.evaluate(
        df_wbc[["norm_center_x", "norm_center_y"]].T
    )
    kde_density_norm = kde_density_wbc / kde_density_wbc.max()

    if option == 0:
        adjustment = base_adjustment + (density_multiplier * kde_density_norm)
    elif option == 1:
        density_percentile = np.percentile(
            kde_density_norm, [percentile_low, percentile_high]
        )
        adjustment = np.ones_like(kde_density_norm)
        adjustment[kde_density_norm < density_percentile[0]] = low_density_adjustment
        adjustment[kde_density_norm > density_percentile[1]] = high_density_adjustment
    elif option == 2:
        log_scale = np.log1p(kde_density_norm)
        adjustment = adjustment_range_low + (log_scale_factor * log_scale)
    elif option == 3:
        high_density_threshold = np.percentile(kde_density_norm, percentile_high)
        low_density_threshold = np.percentile(kde_density_norm, percentile_low)
        high_density_mask = kde_density_norm > high_density_threshold
        low_density_mask = kde_density_norm < low_density_threshold
        adjustment = np.ones_like(kde_density_norm)
        adjustment[high_density_mask] = high_density_adjustment
        adjustment[low_density_mask] = low_density_adjustment
    elif option == 4:
        from scipy.special import expit

        centered_density = kde_density_norm - kde_density_norm.mean()
        smooth_step = expit(centered_density * expit_scale)
        adjustment = adjustment_range_low + (
            (adjustment_range_high - adjustment_range_low) * smooth_step
        )
    else:
        adjustment = np.ones_like(kde_density_norm)

    # Apply adjustment and clip
    df_wbc["confidence"] = np.clip(df_wbc["confidence"] * adjustment, 0, 1)
    df.loc[df["class"] == "WBC", "confidence"] = df_wbc["confidence"].values

    return df


def postprocessing_pipeline(CONFIG, df):
    """Run postprocessing pipeline with the given configuration."""
    # Unpack flags and parameters from CONFIG
    flags = [
        "use_size_adjustment",
        "use_remove_edges",
        "use_spatial_density_troph",
        "use_spatial_density_wbc",
    ]
    params = {key: CONFIG.get(key) for key in CONFIG.keys() if key not in flags}
    print(params)
    # Read initial data
    train_df = pd.read_csv(CONFIG["TRAIN_CSV"])
    # Process bounding boxes
    df = process_df_bbox(df, CONFIG["DATA_DIR"])
    train_df = process_df_bbox(train_df, CONFIG["DATA_DIR"])
    df = basic_postprocess(
        df, CONFIG["DATA_DIR"], CONFIG["NEG_CSV"], CONFIG["TEST_CSV"]
    )

    # Apply processing steps based on flags
    if CONFIG.get("use_size_adjustment", False):
        df = factor_bbox_size_change(
            df, params["size_factor_troph"], params["size_factor_wbc"]
        )
    if CONFIG.get("use_remove_edges", False):
        df = remove_bbox_near_edges(
            df, CONFIG["DATA_DIR"], params["edge_threshold"], params["border_threshold"]
        )

    # Apply spatial density contour adjustments
    for key, fn in [
        ("use_spatial_density_troph", spatial_density_contour_troph),
        ("use_spatial_density_wbc", spatial_density_contour_wbc),
    ]:
        if CONFIG.get(key, False):
            df = fn(
                df,
                train_df,
                CONFIG,
                params[f'option_{key.split("_")[-1]}'],
                base_adjustment=params[f'base_adjustment_{key.split("_")[-1]}'],
                density_multiplier=params[f'density_multiplier_{key.split("_")[-1]}'],
                percentile_low=params[f'percentile_low_{key.split("_")[-1]}'],
                percentile_high=params[f'percentile_high_{key.split("_")[-1]}'],
                low_density_adjustment=params[
                    f'low_density_adjustment_{key.split("_")[-1]}'
                ],
                high_density_adjustment=params[
                    f'high_density_adjustment_{key.split("_")[-1]}'
                ],
                log_scale_factor=params[f'log_scale_factor_{key.split("_")[-1]}'],
                expit_scale=params[f'expit_scale_{key.split("_")[-1]}'],
                adjustment_range_low=params[
                    f'adjustment_range_low_{key.split("_")[-1]}'
                ],
                adjustment_range_high=params[
                    f'adjustment_range_high_{key.split("_")[-1]}'
                ],
            )

    return df[["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"]]


def ensemble_pipeline(
    CONFIG, df_list, weight_list, classes=["Trophozoite", "WBC"]
):  # create flag for dual ensemble
    """Run ensemble pipeline on a list of dataframes. Using nms,wbf or soft-nms. specify conf and iou thresholds+wbf_reduction"""
    form = CONFIG.get("form", "wbf")
    nms_iou_threshold = CONFIG.get("nms_iou_threshold")
    wbf_iou_threshold = CONFIG.get("wbf_iou_threshold")
    wbf_conf_threshold = CONFIG.get("wbf_conf_threshold")
    wbf_reduction = CONFIG.get("wbf_reduction")
    # print(form, nms_iou_threshold, wbf_iou_threshold, wbf_conf_threshold, wbf_reduction)
    if form == "wbf":
        ensemble = DualEnsemble(
            form=form,
            iou_threshold=wbf_iou_threshold,
            classes=classes,
            conf_threshold=wbf_conf_threshold,
            weights=weight_list,
            wbf_reduction=wbf_reduction,
        )
    elif form == "nms":
        ensemble = DualEnsemble(
            form=form,
            iou_threshold=nms_iou_threshold,
            classes=classes,
            conf_threshold=0.0,
        )

    elif form == "soft_nms":
        ensemble = DualEnsemble(
            form=form,
            iou_threshold=nms_iou_threshold,
            classes=classes,
            conf_threshold=0.0,
        )

    df = ensemble(df_list)
    return df


def ensemble_class_specific_pipeline(
    CONFIG, df_list, weight_list, classes=["Trophozoite", "WBC"]
):
    """Run ensemble pipeline with class-specific parameters."""
    # Initialize empty result DataFrame
    final_df = pd.DataFrame()
    # TODO: remove above line

    # Process each class separately
    for j, class_name in enumerate(classes):
        class_key = class_name  # Convert to lowercase for config keys

        # print(CONFIG)
        # Get class-specific parameters
        class_config = {
            "form": CONFIG.get(f"{class_key}_form", "wbf"),
            "nms_iou_threshold": CONFIG.get(f"{class_key}_nms_iou_threshold"),
            "voting_iou_threshold": CONFIG.get(f"{class_key}_voting_iou_threshold"),
            "soft_nms_iou_threshold": CONFIG.get(f"{class_key}_soft_nms_iou_threshold"),
            "wbf_iou_threshold": CONFIG.get(f"{class_key}_wbf_iou_threshold"),
            "wbf_conf_threshold": CONFIG.get(f"{class_key}_wbf_conf_threshold"),
            "wbf_reduction": CONFIG.get(f"{class_key}_wbf_reduction"),
        }
        # print(df_list[0].head())
        # Filter DataFrames for current class
        class_dfs = [df[df["class"] == class_name].copy() for df in df_list]
        class_dfs = [df[df["confidence"] > 1e-3] for df in class_dfs]
        # print(class_config)
        # Create class-specific ensemble
        if class_config["form"] == "wbf":
            ensemble = DualEnsemble(
                form="wbf",
                iou_threshold=class_config["wbf_iou_threshold"],
                classes=[class_name],
                conf_threshold=class_config["wbf_conf_threshold"],
                weights=weight_list[j],
                wbf_reduction=class_config["wbf_reduction"],
            )
        elif class_config["form"] == "nms":
            ensemble = DualEnsemble(
                form="nms",
                iou_threshold=class_config["nms_iou_threshold"],
                classes=[class_name],
                conf_threshold=0.001,
            )
        elif class_config["form"] == "voting":
            ensemble = DualEnsemble(
                form="voting",
                iou_threshold=class_config["voting_iou_threshold"],
                classes=[class_name],
                conf_threshold=0.1,
            )
        elif class_config["form"] == "soft_nms":
            ensemble = DualEnsemble(
                form="soft_nms",
                iou_threshold=class_config["soft_nms_iou_threshold"],
                classes=[class_name],
                conf_threshold=0.001,
            )

        # Process class predictions
        class_result = ensemble(class_dfs)

        # Combine results
        final_df = pd.concat([final_df, class_result], ignore_index=True)

    return final_df
