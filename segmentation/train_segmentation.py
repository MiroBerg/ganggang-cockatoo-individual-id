import os
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm
import torch

print("CUDA available:", torch.cuda.is_available())

# Path to YOLO dataset yaml file
config = "...path_to.../config.yaml"  

# Load YOLO model
seg_model = YOLO("...path_to.../yolo11l-seg.pt")

# Set model save directory
save_dir = "...path_to.../runs"
os.makedirs(save_dir, exist_ok=True)

# Train YOLO model
seg_model.train(
    data=config,         # Path to the dataset configuration file
    epochs=50,           # Number of training epochs
    batch=8,             # Batch size
    imgsz=640,           # Image size for training
    lr0=0.0005,          # Initial learning rate
    optimizer="AdamW",   # Optimizer
    weight_decay=0.0005, # Weight decay
    patience=10,         # Epochs before training is stopped, when no improvement
    freeze=0,            # Number of frozen layers
    warmup_epochs=3,     # Number of warm up epochs
    name="fine_tuned2",  # Name of model
    project=save_dir     # Set save directory
)
