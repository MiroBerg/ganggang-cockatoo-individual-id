from ultralytics import YOLO
import cv2
import glob
import os
import torch

# Set path to trained model
file_path = "...path_to.../weights/best.pt"

# Check if the file exists and is accessible
if os.access(file_path, os.R_OK):
    print("File is readable.")
else:
    print("File is not readable or doesn't exist.")

# Load trained YOLO model
seg_model = YOLO(file_path)

# Set save directory
save_dir = "...path_to.../runs"
os.makedirs(save_dir, exist_ok=True)

# Automatically evaluate model on validation data
results = seg_model.val(project=save_dir)

seg_model.predict(
    source = "...path_to...", # Path to validation dataset folder
    save=True,                # Set save images to true
    save_txt=False,           # Set save txt to false
    save_conf=True,           # Set save confidence to true
    imgsz=640,                # Set input image size
    project=save_dir          # Set save directory
)
