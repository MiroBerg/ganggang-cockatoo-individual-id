import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

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

# Import data
label_csv_anu = pd.read_csv("...path_to.../anu_individuals_csv.csv") # csv of labelled ANU dataset
label_csv_cook = pd.read_csv("...path_to.../cook_individuals_csv.csv") # csv of labelled COOK dataset
label_csv = pd.concat([label_csv_anu, label_csv_cook], ignore_index=True) # Add datasets together

# Define training dataframe
train_data = label_csv[label_csv["val"] == 0]
print("Training Images:", len(train_data["img_name"]))
print("Validation Images:", len(val_data["img_name"]))

# Create column with perspective + individual (e.g. "r_Abby")
train_data["persp_ind"] = train_data["persp_label"] + "_" + train_data["individual_1"]

# Balance training dataset
train_ind_dict = dict(Counter(train_data["persp_ind"].tolist()))
def_num_pic = max(train_ind_dict.values()) # Set image count to absolute maximum

# Oversample
for persp_ind in train_ind_dict:
    perspective = persp_ind[0]
    individual = persp_ind[2:]
    num_sample = def_num_pic - train_ind_dict[persp_ind]
    filtered_df = train_data[train_data['individual_1'] == individual]
    filtered_df = filtered_df[filtered_df['persp_label'] == perspective]
    sampled_rows = filtered_df.sample(n = num_sample, random_state = 10, replace=True)
    train_data = pd.concat([train_data, sampled_rows])

# Add column with number instead of perspective
ind_dict_path = "...path_to.../ind_dict.txt"
ind_dict = {}
with open(ind_dict_path, "r") as f:
    for line in f:
        key, value = line.strip().split(',')
        key = key.strip()
        value = int(value.strip())
        ind_dict[key] = value

for index, row in train_data.iterrows():
    train_data.loc[index, "ind_num"] = str(ind_dict[str(row["individual_1"])])

for index, row in val_data.iterrows():
    val_data.loc[index, "ind_num"] = str(ind_dict[str(row["individual_1"])])
