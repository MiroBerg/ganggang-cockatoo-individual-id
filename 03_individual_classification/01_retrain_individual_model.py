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
def_num_pic = max(train_ind_dict.values()) # Set number to maximum within perspective
# Oversample
for individual in train_ind_dict:
    num_sample = def_num_pic - train_ind_dict[individual]
    filtered_df = train_data[train_data['individual_1'] == individual]
    sampled_rows = filtered_df.sample(n = num_sample, random_state = 10, replace=True)
    train_data = pd.concat([train_data, sampled_rows])

# Balance validation dataset
val_ind_dict = dict(Counter(val_data["individual_1"].tolist()))
def_num_pic = min(val_ind_dict.values()) # Set number to minimum (which is always 20)
# Undersample
for individual in val_ind_dict:
    num_sample = val_ind_dict[individual] - def_num_pic
    if num_sample > 0:
        filtered_df = val_data[val_data['individual_1'] == individual]
        sampled_rows = filtered_df.sample(n=num_sample, random_state=10)
        val_data = val_data.drop(sampled_rows.index)

# Remove all inividuals from training, which are not in the validation dataset
val_ind = list(val_ind_dict.keys())
train_data_only_val = train_data[train_data["individual_1"].isin(val_ind)]

# Add column with number instead of perspective
ind_dict_path = "...path_to.../ind_dict.txt"
ind_dict = {}
with open(ind_dict_path, "r") as f:
    for line in f:
        key, value = line.strip().split(',')
        key = key.strip()
        value = int(value.strip())
        ind_dict[key] = value
for index, row in train_data_only_val.iterrows():
    train_data_only_val.loc[index, "ind_num"] = str(ind_dict[str(row["individual_1"])])
for index, row in val_data.iterrows():
    val_data.loc[index, "ind_num"] = str(ind_dict[str(row["individual_1"])])

# Training image generator
train_generator_onlyval = ImageDataGenerator(
    brightness_range=[0.85, 1.15],
    zoom_range=0.1
)
train_generator_data_onlyval = train_generator_onlyval.flow_from_dataframe(
    train_data_only_val,
    x_col="file_path",
    y_col="ind_num",
    target_size = (384, 384),
    batch_size=32,
    shuffle=True,
    class_mode="sparse",
    color_mode="rgb",
    preprocessing_function = preprocess_input)

# Validation image generator
val_generator = ImageDataGenerator()
val_generator_data=val_generator.flow_from_dataframe(val_data,
                                x_col="file_path",
                                y_col="ind_num",
                                target_size = (384, 384),
                                batch_size=8,
                                shuffle=False,
                                class_mode="sparse",
                                color_mode="rgb",
                                preprocessing_function = preprocess_input)

# Load model
backbone = load_model("...path_to.../individual_backbone_model.keras")

# Define architecture
x = backbone.get_layer("global_average_pooling2d").output # Take base from backbone model
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(train_data_only_val["ind_num"].nunique(), activation="softmax")(x)
model = Model(inputs=backbone.input, outputs=outputs)

# Training only classification head for 3 epochs
for layer in backbone.layers:
    layer.trainable = False
    
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['acc'])

model.fit(train_generator_data_onlyval,
          steps_per_epoch=len(train_generator_data_onlyval),
          epochs=3,
          validation_data=val_generator_data, 
          validation_steps=len(val_generator_data))

# Training with top 30 layers unfrozen
for layer in backbone.layers[-30:]:
    layer.trainable = True

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['acc'])

filepath = "...path_to.../final_individual_model.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping_monitor = EarlyStopping(monitor = "val_loss", patience=5, restore_best_weights=True)
callbacks_list = [checkpoint, early_stopping_monitor]

history = model.fit(train_generator_data_onlyval,
          steps_per_epoch=len(train_generator_data_onlyval),
          epochs=50,
          validation_data=val_generator_data, 
          validation_steps=len(val_generator_data),
          callbacks=callbacks_list)
