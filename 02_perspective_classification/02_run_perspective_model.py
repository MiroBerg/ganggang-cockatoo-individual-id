from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# Load ANU
anu_csv = pd.read_csv("...path_to.../anu_csv.csv")

# Add new columns for the perspective and confidence
anu_csv["persp_label"] = ""
anu_csv["persp_conf"] = ""

# Import model
persp_model = load_model("...path_to.../persp_model.keras")

# Load image size of the model
_, height, width, channels = persp_model.input_shape
target_size = (width, height)

# Set perspective labels
labels = ["f", "r", "l", "b"]

# Preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size = target_size) # Load image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img) # Apply preprocessing functoin
    return img

# Run perspective model for each row
def run_perspective(row):
    if row["seg_conf"] not in [None, ""] and float(row["seg_conf"]) >= 0.8:
        file_path = "...path_to.../image_folder/" + row["img_name"] + ".JPG"
        img = preprocess_image(file_path)
        preds = persp_model.predict(img, verbose=0)
        preds = preds[0] # Remove batch dimension
        
        top_index = np.argmax(preds)  # Get index of highest predicted class
        top_conf = preds[top_index]   # Get top confidence of that class
        top_label = labels[top_index]
        
        row["persp_label"] = top_label
        row["persp_conf"] = top_conf
    return row

tqdm.pandas(desc="Processing Images") # Set pandas tqdm title
anu_csv = anu_csv.progress_apply(run_perspective, axis=1) # Apply function to all rows

anu_csv.to_csv("...path_to.../persp_csv.csv", index=False)
