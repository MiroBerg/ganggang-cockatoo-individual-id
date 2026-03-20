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

# Check if GPU is detected
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load label csv
persp_labels = pd.read_csv("...path_to.../perspective_labels.csv")

# Create dataframes for training and validation
train_data = persp_labels[persp_labels["val"] == 0]
val_data = persp_labels[persp_labels["val"] == 1]
print("Training Images:", len(train_data["img_name"]))
print("Validation Images:", len(val_data["img_name"]))

# Define number of pictures per category in training
def_num_pic = 600
# Undersample Training
for perspective in train_persp_dict:
    num_sample = train_persp_dict[perspective] - def_num_pic
    if num_sample > 0:
        filtered_df = train_data[train_data["true_persp"] == perspective]
        sampled_rows = filtered_df.sample(n=num_sample, random_state=10)
        train_data = train_data.drop(sampled_rows.index)
# Oversample Training
for perspective in train_persp_dict:
    num_sample = def_num_pic - train_persp_dict[perspective]
    filtered_df = train_data[train_data["true_persp"] == perspective]
    sampled_rows = filtered_df.sample(n = num_sample, random_state = 10, replace=True)
    train_data = pd.concat([train_data, sampled_rows])
  
# Define number of pictures per category in validation
def_num_pic = 46
# Undersample Validation
for perspective in val_persp_dict:
    num_sample = val_persp_dict[perspective] - def_num_pic
    if num_sample > 0:
        filtered_df = val_data[val_data["true_persp"] == perspective]
        sampled_rows = filtered_df.sample(n=num_sample, random_state=10)
        val_data = val_data.drop(sampled_rows.index)

# Add column with number instead of perspective
persp_dict = {"f": 0, "r": 1, "l": 2, "b": 3}
train_data["persp_num"] = ""
val_data["persp_num"] = ""
for index, row in train_data.iterrows():
    train_data.loc[index, "persp_num"] = str(persp_dict[str(row["true_persp"])])
for index, row in val_data.iterrows():
    val_data.loc[index, "persp_num"] = str(persp_dict[str(row["true_persp"])])

# Create training image generator
train_generator = ImageDataGenerator()
train_generator_data=train_generator.flow_from_dataframe(train_data,
                                x_col="file_path",
                                y_col="persp_num",
                                target_size = (224, 224),
                                rotation_range=90,
                                zoom_range=0.4,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                batch_size=4,
                                shuffle=True,
                                class_mode="sparse",
                                color_mode="rgb",
                                preprocessing_function = preprocess_input)


# Create validation image generator
val_generator = ImageDataGenerator()
val_generator_data=val_generator.flow_from_dataframe(val_data,
                                x_col="file_path",
                                y_col="persp_num",
                                target_size = (224, 224),
                                batch_size=4,
                                shuffle=False,
                                class_mode="sparse",
                                color_mode="rgb",
                                preprocessing_function = preprocess_input)

# Create model architecture
base = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictors = Dense(4, activation='softmax')(x)
model = Model(inputs=base.input, outputs=predictors)

filepath = "...path_to.../perspective_model.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping_monitor = EarlyStopping(patience=10)
callbacks_list = [checkpoint, early_stopping_monitor]

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['acc'])

# Train model
history = model.fit(train_generator_data,
          steps_per_epoch=len(train_generator_data),
          epochs=100,
          validation_data=val_generator_data, 
          validation_steps=len(val_generator_data),
          callbacks=callbacks_list)
