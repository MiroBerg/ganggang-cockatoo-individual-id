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
label_csv = pd.read_csv("/home/aplin-ai-1/Documents/final/05_individuals_allpersp_cook_noleak/csv_files/label_csv_anu_noleak.csv")
label_csv = label_csv[label_csv["persp_label"] == "b"]
train_data = label_csv[label_csv["val"] == 0]
val_data = label_csv[label_csv["val"] == 1]
test_data = label_csv[label_csv["val"] == 2]

print("Training Images:", len(train_data["img_name"]))
print("Validation Images:", len(val_data["img_name"]))
print("Testing Images:", len(test_data["img_name"]))
