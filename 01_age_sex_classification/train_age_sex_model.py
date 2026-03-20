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
train_data = pd.read_csv("...path_to.../training_dataset.csv")
val_data = pd.read_csv("...path_to.../validation_dataset.csv")

print("Training Images:", len(train_data["img_name"]))
print("Validation Images:", len(val_data["img_name"]))

# Add column with number instead of perspective
as_dict = {"a_f": 0, "a_m": 1, "j_f": 2, "j_m": 3}
for index, row in train_data.iterrows():
    train_data.loc[index, "as_num"] = str(as_dict[str(row["age_sex"])])
for index, row in val_data.iterrows():
    val_data.loc[index, "as_num"] = str(as_dict[str(row["age_sex"])])
for index, row in test_data.iterrows():
    test_data.loc[index, "as_num"] = str(as_dict[str(row["age_sex"])])

# Create training generator
train_generator = ImageDataGenerator(
    brightness_range=[0.85, 1.15],
    zoom_range=0.1
)
train_generator_data = train_generator.flow_from_dataframe(
    train_data,
    x_col="file_path",
    y_col="as_num",
    target_size = (384, 384),
                                                         
    batch_size=32,
    shuffle=True,
    class_mode="sparse",
    color_mode="rgb",
    preprocessing_function = preprocess_input)

# Create validation generator
val_generator = ImageDataGenerator()
val_generator_data=val_generator.flow_from_dataframe(val_data,
                                x_col="file_path",
                                y_col="as_num",
                                target_size = (384, 384),
                                batch_size=8,
                                shuffle=False,
                                class_mode="sparse",
                                color_mode="rgb",
                                preprocessing_function = preprocess_input)

# Create model architecture
base = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(384, 384, 3)) # Input shape hsame as generators
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation="softmax")(x)

# Freeze layers = 0
for layer in base.layers:
    layer.trainable = True

# Create model
model = Model(inputs=base.input, outputs=outputs)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=1e-5),
              metrics=['acc'])

filepath = "...path_to.../age_sex_model.keras" # Set path to final model save
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') # Save model only when validation loss improves
early_stopping_monitor = EarlyStopping(monitor = "val_loss", patience=5, restore_best_weights=True) # Stop training when no improvement after 5 epochs
callbacks_list = [checkpoint, early_stopping_monitor] # Implement above

# Train model
history = model.fit(train_generator_data,
          steps_per_epoch=len(train_generator_data),
          epochs=50,
          validation_data=val_generator_data, 
          validation_steps=len(val_generator_data),
          callbacks=callbacks_list)
