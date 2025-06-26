# run_resnet50_model.py
import os
import time
import datetime
import io

import tensorflow as tf
import keras_cv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Configuration ---
TRAIN_DATA_DIR = '/workspace/project/wastedata_split/train'
VALIDATION_DATA_DIR = '/workspace/project/wastedata_split/val'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 64
NUM_CLASSES = 10
INITIAL_EPOCHS = 10
FINETUNE_EPOCHS = 20
TOTAL_EPOCHS = INITIAL_EPOCHS + FINETUNE_EPOCHS

# --- 2. Data Pipeline ---
print("Preparing tf.data.Dataset pipeline for ResNet50...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR, label_mode='int', seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=True)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DATA_DIR, label_mode='int', seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False)
class_names = train_ds.class_names

def resnet_preprocess(images, labels):
    return preprocess_input(images), labels

# Apply ResNet50's specific preprocessing.
train_ds = train_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. Build ResNet50 Model ---
print("\nBuilding ResNet50 Transfer Learning Model...")
base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. TensorBoard Setup ---
log_dir = "logs/resnet50_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# --- 5. Two-Phase Training ---
# Phase 1: Feature Extraction
base_model.trainable = False
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\n--- Phase 1: Training Head ---")
history = model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds, callbacks=[tensorboard_callback])

# Phase 2: Fine-Tuning
base_model.trainable = True
fine_tune_at = 143
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\n--- Phase 2: Fine-Tuning ---")
history_fine = model.fit(train_ds, epochs=TOTAL_EPOCHS, initial_epoch=history.epoch[-1], validation_data=val_ds, callbacks=[tensorboard_callback, early_stopping_callback])

print("\nEvaluating Model...")

# Plotting training history
acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']
loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 6))
# ... (plotting code is the same) ...
plt.savefig("custom_model_final_history.png")
plt.show()

# Classification Report and Confusion Matrix
print("\nGenerating Classification Report and Confusion Matrix...")
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)

print('\nClassification Report')
print(classification_report(y_true, y_pred, target_names=class_names))

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
# ... (confusion matrix plotting code is the same) ...
plt.savefig("custom_model_final_cm.png")
plt.show()

# Save the final model