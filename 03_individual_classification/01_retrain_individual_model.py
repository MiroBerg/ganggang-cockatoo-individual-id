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

# Create dataframes for training and validation
label_csv = pd.read_csv("...path_to.../individual_label_csv.csv")

# !!! Adapt this line depending on perspective ("b":back, "l": left, "r": right, "f": front)
label_csv = label_csv[label_csv["persp_label"] == "b"]

# Define training and validation dataframes
train_data = label_csv[label_csv["val"] == 0]
val_data = label_csv[label_csv["val"] == 1]

print("Training Images:", len(train_data["img_name"]))
print("Validation Images:", len(val_data["img_name"]))

# Balance training dataset
train_ind_dict = dict(Counter(train_data["individual_1"].tolist()))
def_num_pic = max(train_ind_dict.values()) # Set number to maximum overall

# Oversample
for individual in train_ind_dict:
    num_sample = def_num_pic - train_ind_dict[individual]
    filtered_df = train_data[train_data['individual_1'] == individual]
    sampled_rows = filtered_df.sample(n = num_sample, random_state = 10, replace=True)
    train_data = pd.concat([train_data, sampled_rows])

# Check for balanced data
train_ind_dict = dict(Counter(train_data["individual_1"].tolist()))
train_ind_dict
