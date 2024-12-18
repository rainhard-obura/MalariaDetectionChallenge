import os
import shutil

import cv2
import yaml
from tqdm import tqdm


def save_dataset_in_yolo(image_folder, preprocessed_df, output_dir):
    """
    Converts a dataset into YOLO format and saves it to disk, excluding NEG class images.
    Also generates a dataset.yaml file for YOLO configuration.

    Args:
        image_folder (str): Path to folder containing source images
        preprocessed_df (pd.DataFrame): DataFrame containing annotations with columns:
            ['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax']
        output_dir (str): Path where the YOLO dataset will be saved

    Returns:
        dict: Statistics about the processed dataset
    """
    # remove output dir
    shutil.rmtree(output_dir, ignore_errors=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=False)

    # Clear existing directory if it exists
    shutil.rmtree(os.path.join(output_dir, "train"), ignore_errors=True)

    # Remove NEG class images
    pos_df = preprocessed_df[preprocessed_df["class"] != "NEG"]

    # Get unique positive classes and create mapping (excluding NEG)
    unique_classes = sorted(pos_df["class"].unique())
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}

    # Get unique image IDs (excluding images that only have NEG classes)
    image_ids = pos_df["Image_ID"].unique()

    # Create directories
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)

    processed_count = 0
    skipped_count = 0

    for img_name in tqdm(image_ids, desc="Processing images"):
        # Load image
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read image {img_path} - Skipping.")
            skipped_count += 1
            continue

        imheight, imwidth, _ = img.shape

        # Get annotations for this image (excluding NEG class)
        img_boxes = pos_df[pos_df["Image_ID"] == img_name]

        # Create label file
        label_filename = f"{''.join(img_name.split('.')[:-1])}.txt"
        label_path = os.path.join(train_dir, "labels", label_filename)

        with open(label_path, "w+") as f:
            for _, row in img_boxes.iterrows():
                # Calculate normalized box dimensions
                x_center = (row["xmin"] + row["xmax"]) / 2 / imwidth
                y_center = (row["ymin"] + row["ymax"]) / 2 / imheight
                width = (row["xmax"] - row["xmin"]) / imwidth
                height = (row["ymax"] - row["ymin"]) / imheight

                # Write YOLO format line
                class_id = class_mapping[row["class"]]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Save image
        cv2.imwrite(os.path.join(train_dir, "images", img_name), img)
        processed_count += 1

    # Save class mapping
    with open(os.path.join(output_dir, "classes.txt"), "w+") as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")

    # Generate dataset.yaml
    yaml_content = {
        "names": {class_mapping[k]: k for k in class_mapping},
        "path": os.path.abspath(output_dir),
        "train": "train",
        "val": "train",
    }

    # Save dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # Compile statistics
    stats = {
        "total_images_before_filtering": len(preprocessed_df["Image_ID"].unique()),
        "total_images_after_filtering": len(image_ids),
        "processed": processed_count,
        "skipped": skipped_count,
        "class_mapping": class_mapping,
        "classes": unique_classes,
    }

    print("\nDataset generation complete!")
    print(f"Total images before filtering: {stats['total_images_before_filtering']}")
    print(
        f"Total images after removing NEG class: {stats['total_images_after_filtering']}"
    )
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Classes: {', '.join(stats['classes'])}")
    print(f"Dataset YAML file saved to: {yaml_path}")

    return stats
