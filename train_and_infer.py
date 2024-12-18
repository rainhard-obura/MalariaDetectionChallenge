import glob
import os
from time import time

import pandas as pd

from inference.detr.inference_detr_models import detr_predict_tta
from inference.inference_neg_model import inference
from inference.yolo.inference_yolo_models import yolo_predict_tta
from postprocess import run_postprocessing
from preprocessing.dataset_yolo_format import save_dataset_in_yolo
from preprocessing.filter_boxes import get_filtered_train_df
from training.detr.train_detr_models import get_trained_detr_models
from training.train_neg_model import train_model
from training.yolo.train_yolo_models import get_trained_yolo_models
from util.save import add_negs_to_submission
from util.yaml_structuring import create_structured_config, load_yaml_config

train_start = time()
# --- PREPROCESSING ---
train_df = pd.read_csv("data/csv_files/Train.csv")

# Remove double boxes from the labels
filtered_train_df = get_filtered_train_df(train_df, iou_threshold=0.3)

# Generate the YOLO dataset, ignoring the NEG labels
save_dataset_in_yolo("data/img", filtered_train_df, "data/yolo_ds")

# --- TRAINING ---
yolo_models = get_trained_yolo_models(
    glob.glob("config_files/yolo_train_config_files/*.yaml"),
    "data/yolo_ds/dataset.yaml",
)
detr_models = get_trained_detr_models(
    glob.glob("config_files/detr_train_config_files/*.yaml"),
    "data/yolo_ds/dataset.yaml",
)
os.makedirs("models", exist_ok=True)
neg_model = train_model("data/img", "data/csv_files/Train.csv", 2)

train_time = time() - train_start
print(f"Training took {train_time / 3600:.2f} hours")

inference_start = time()
# --- INFERENCE ---
# save our NEG predictions in a csv
result = inference(neg_model, "data/img", "data/csv_files/Test.csv")
neg_preds = []
for id, pred in result.items():
    neg_preds.append([id, pred.replace("POS", "NON_NEG")])
pd.DataFrame(neg_preds, columns=["Image_ID", "class"]).to_csv(
    "data/csv_files/NEG_OR_NOT.csv", index=False
)

test_df = pd.read_csv("data/csv_files/Test.csv")
test_images_paths = glob.glob("data/img/*.jpg")
test_image_ids = [os.path.basename(img) for img in test_images_paths]
test_images_paths = [
    img
    for img in test_images_paths
    if os.path.basename(img) in test_df["Image_ID"].values
]

final_preds: dict[str, list[pd.DataFrame]] = {}

# Get predictions for YOLO using TTA
for model in yolo_models:
    yolo_preds = yolo_predict_tta(model, img_paths=test_images_paths)
    final_preds[f"yolo"] = yolo_preds

# Get predictions for DETR using TTA
detr_config_files = glob.glob("config_files/detr_train_config_files/*.yaml")
for model, config_file in zip(detr_models, detr_config_files):
    detr_preds = detr_predict_tta(model, config_file, img_paths=test_images_paths)
    final_preds["detr"] = detr_preds

# Save the final predictions to disk
os.makedirs("data/predictions", exist_ok=True)
tta_files = []
for model_name, preds in final_preds.items():
    for i, df in enumerate(preds):
        csv_name = f"data/predictions/{model_name[:3]}_predictions_{i + 1}.csv"
        df.to_csv(csv_name, index=False)
        tta_files.append(csv_name)

# --- POSTPROCESSING ---
config_file = "parameters/postprocessing_config_files/distinctive_sweep_169.yaml"
config = load_yaml_config(config_file)
param_config = create_structured_config(config["parameters"])

detr_tta_files = [f for f in tta_files if "det" in f]
yolo_tta_files = [f for f in tta_files if "yol" in f]

all_df = run_postprocessing(param_config, 1, yolo_tta_files, detr_tta_files)

# Add NEG predictions to the final submission
os.makedirs("submissions", exist_ok=True)
final_submission_df = add_negs_to_submission(
    df=all_df,
    neg_csv="data/csv_files/NEG_OR_NOT.csv",
    test_csv="data/csv_files/Test.csv",
)
final_submission_df.to_csv("submissions/final_submission.csv", index=False)

inference_time = time() - inference_start
print(f"Inference took {inference_time / 3600:.2f} hours")
