# train_gpu_augmented.py
import os
import tensorflow as tf
import keras_cv

# This script should be run like this to suppress logs:
# TF_CPP_MIN_LOG_LEVEL='2' python train_gpu_augmented.py

# --- In-script check to confirm GPU detection ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! TensorFlow has found and is using the following GPU(s): {gpus}")
else:
    print("❌ Error! TensorFlow did NOT find any GPUs.")


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
DATA_DIR = '/workspace/project/data/dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64 # We can often use a larger batch size with an optimized pipeline
NUM_CLASSES = 6
EPOCHS = 30

# --- 1. KerasCV Augmentation Pipeline ---
# Define a sequence of augmentation layers that will run on the GPU.
# These values mirror the old ImageDataGenerator settings.
augmenter = keras_cv.layers.RandomAugmentationPipeline(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal"),
        keras_cv.layers.RandomRotation(factor=0.1), # factor of 0.1 is approx 36 degrees
        keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2),
        keras_cv.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        keras_cv.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
    ],
    augmentations_per_image=1,
)

# --- 2. High-Performance Data Loading with tf.data ---
print("Preparing tf.data.Dataset pipeline...")

# Load the training data from disk
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Load the validation data from disk
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Create a function to apply the augmentations and preprocessing
# Note: Augmentations are applied BEFORE the model's specific preprocessing
def apply_augmentations_and_preprocessing(images, labels):
    images = augmenter(images)
    images = preprocess_input(images) # Use MobileNetV2's required preprocessing
    return images, labels

# Apply the augmentations to the training set
train_ds = train_ds.map(apply_augmentations_and_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)

# Apply only the preprocessing to the validation set
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Configure the datasets for peak performance
# .prefetch() overlaps data preprocessing and model execution, eliminating bottlenecks.
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# --- 3. Model Definition (Same as before) ---
print("\nBuilding model with MobileNetV2 base...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False # Freeze the base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# --- 4. Model Training ---
# The model.fit call is now much simpler. We don't need workers, steps, etc.
# tf.data handles all the performance details automatically.
print("\nStarting Training with GPU-accelerated pipeline...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# --- 5. Final Model Evaluation ---
print("\nEvaluating Model...")
# (Evaluation code for plotting, classification report, etc., would go here, same as before)
# ...