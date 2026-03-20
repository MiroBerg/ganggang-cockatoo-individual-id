import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

# Import ANU single birds csv
anu_csv = pd.read_csv("...path_to.../csv_files/anu_birds.csv")

# Add new columns for the segmentation and confidence
single_anu_csv["seg_poly"] = ""
single_anu_csv["seg_conf"] = ""

# Load segmentation model
seg_model = YOLO("...path_to.../model.pt")

# Define function to predict segmentation
def predict_seg(row):
    if row["individual_1"] != "bq": # Only use images of high quality
        result = seg_model.predict(source = row["file_path"], verbose = False)[0] # Make model prediction (verbose=False is just that it doesn't log to console)

        # Skip if no detections
        if not result.boxes:
            row["seg_poly"] = "nd"
            row["seg_conf"] = 0
            return row

        confs = result.boxes.conf.detach().cpu().numpy() # Get confidence scores
        polygons = result.masks.xy # Get Polygons
        
        best_index = np.argmax(confs) # Find index with highest confidence score
        
        best_conf = confs[best_index] # Save highest confidence
        best_poly = polygons[best_index].tolist() # Save polygon with highest confidence

        row["seg_poly"] = json.dumps(best_poly) # Save best polygon as json (use json.loads!!)
        row["seg_conf"] = best_conf # Save best confidence

    return row

tqdm.pandas(desc="Processing Images") # Set pandas tqdm title
anu_csv = anu_csv.progress_apply(predict_seg, axis=1) # Apply function to all rows

# Set output folder
output_folder = "...path_to.../dataset/cropped/"

# Define function to extract bird
def extract_bird(row):
    if row["seg_conf"] not in [None, ""] and float(row["seg_conf"]) >= 0.8: # Check that confidence is not na and >0.8
        img = cv2.imread(row["file_path"]) # Load image
        polygon = np.array(json.loads(row["seg_poly"]), dtype=np.int32) # Load segmentation polygon

        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Create mask with size of the image
        cv2.fillPoly(mask, [polygon], 255)  # Fill the polygon with white

        masked_img = cv2.copyTo(img, mask) # Apply segementation mask to image

        x, y, w, h = cv2.boundingRect(polygon) # Extract bounding rectangle
        cropped_img = masked_img[y:y+h, x:x+w] # Crop image to bounding rectangle

        output_path = output_folder + row["img_name"] + ".JPG" # Define output path
        cv2.imwrite(output_path, cropped_img) # Create extracted bird image

tqdm.pandas(desc="Segmenting Images") # Set pandas tqdm title
anu_csv = anu_csv.progress_apply(extract_bird, axis=1) # Apply function to all rows

# Save as csv
anu_csv.to_csv("...path_to.../anu_birds_segmented.csv", index=False)

