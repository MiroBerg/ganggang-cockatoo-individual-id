import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Check if GPU is detected
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Create dataframes for testing
label_csv = pd.read_csv("...path_to.../individual_label_csv.csv")

# !!! Adapt this line depending on perspective ("b":back, "l": left, "r": right, "f": front)
label_csv = label_csv[label_csv["persp_label"] == "b"]

# Define training and validation dataframes
test_data = label_csv[label_csv["val"] == 2]
print("Testing Images:", len(test_data["img_name"]))

# Balance testing dataset
test_ind_dict = dict(Counter(test_data["individual_1"].tolist()))
def_num_pic = min(test_ind_dict.values()) # Set number to minimum, which is always 20
# Undersample
for individual in test_ind_dict:
    num_sample = test_ind_dict[individual] - def_num_pic
    if num_sample > 0:
        filtered_df = test_data[test_data['individual_1'] == individual]
        sampled_rows = filtered_df.sample(n=num_sample, random_state=10)
        test_data = test_data.drop(sampled_rows.index)

# Testing image generator
test_generator = ImageDataGenerator()
test_generator_data=test_generator.flow_from_dataframe(test_data,
                                x_col="file_path",
                                y_col="ind_num",
                                target_size = (384, 384),
                                batch_size=8,
                                shuffle=False,
                                class_mode="sparse",
                                color_mode="rgb",
                                preprocessing_function = preprocess_input)

# Function to get true mapping of model values
def get_class_mapping(generator):
    mapping = generator.class_indices.copy()
        mapping = {v: k for k, v in mapping.items()}
    return mapping
test_mapping = get_class_mapping(test_generator_data)

# Load model
model = load_model("...path_to.../final_individual_model.keras")

predicted_label = []
true_label = []
top3_correct = []

for step in range(len(test_generator_data)):
    X, y = next(test_generator_data)
    probs = model.predict(X, verbose=0)
    for i in range(len(probs)):
        true = int(y[i])
        preds = probs[i]
      
        top1 = np.argmax(preds) # Top-1 prediction
      
        top3 = np.argsort(preds)[-3:][::-1]
        top3_correct.append(1 if true in top3 else 0) # Top-3 prediction

        true = int(test_mapping[true]) # Convert to true mapping
        top1 = int(test_mapping[int(top1)]) # Convert to true mapping
        
        predicted_label.append(top1)
        true_label.append(true)

# Show confusion matrix
confusion = confusion_matrix(true_label, predicted_label)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=test_ind_list,
            yticklabels=test_ind_list)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Show classificatoin report
print("Classification Report:")
print(classification_report(true_label,
                            predicted_label,
                            target_names=test_ind_list,
                            digits=3))
print(f"Top-3 Accuracy: {np.mean(top3_correct):.3f}")
