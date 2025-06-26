# run_custom_model_gap.py
import os
import time
import datetime
import io

import tensorflow as tf
import keras_cv

# --- Suppress TensorFlow logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Import Keras Layers ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

# --- Import Other Libraries ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# --- 1. Configuration ---
# IMPORTANT: This should point to the split dataset directory
TRAIN_DATA_DIR = '/workspace/project/wastedata_split/train'
VALIDATION_DATA_DIR = '/workspace/project/wastedata_split/val'

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHS = 10

# --- 2. KerasCV Augmentation Pipeline (GPU Accelerated) ---
print("Defining KerasCV GPU-accelerated augmentation pipeline...")
augmenter = keras_cv.layers.RandomAugmentationPipeline(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal"),
        keras_cv.layers.RandomRotation(factor=0.1),
        keras_cv.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    augmentations_per_image=1,
)

# --- 3. High-Performance Data Loading with tf.data ---
print("Preparing tf.data.Dataset pipeline from explicit train/val folders...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    label_mode='int',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DATA_DIR,
    label_mode='int',
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = train_ds.class_names
print(f"Found classes: {class_names}")

# --- 4. Prepare data pipeline for training ---
def apply_augmentations(images, labels):
    return augmenter(images, training=True), labels

train_ds = train_ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 5. Build the Custom CNN Model (with Efficient GAP Head) ---
print("\nBuilding Custom Model with an Efficient GAP Head...")
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = Rescaling(1./255)(inputs)

# Convolutional Base (Identical to the previous model)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# ===================================================================
#                ### NEW EFFICIENT CLASSIFIER HEAD ###
# This replaces the Flatten -> Dense(512) -> BN block
# ===================================================================
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Use a strong Dropout for regularization
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
# ===================================================================

model = Model(inputs=inputs, outputs=outputs)

# --- 6. Compile and Visualize the Model ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

try:
    plot_model(model, to_file='custom_model_gap_architecture.png', show_shapes=True, show_layer_activations=True)
    print("\n✅ Model architecture plot saved to custom_model_gap_architecture.png\n")
except Exception as e:
    print(f"\n❌ Could not create plot. Ensure graphviz and pydot are installed. Error: {e}")

model.summary() # Note how drastically the parameter count has dropped!

# --- 7. Train the Model ---
# No need for class weights here, as the GAP head is a strong regularizer.
# We can add them back if we see signs of imbalance issues.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

print("\nStarting Training...")
start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping_callback]
)
end_time = time.time()
print(f"\nTotal training time: {(end_time - start_time)/60:.2f} minutes")

# --- 8. Evaluate the Model & Save Artifacts ---
print("\nEvaluating Model...")
# (Full evaluation code for plotting, reports, etc.)
# ...
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('GAP Model - Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('GAP Model - Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("custom_model_gap_history.png")
plt.show()

print("\nGenerating Classification Report and Confusion Matrix...")
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)

print('\nClassification Report')
print(classification_report(y_true, y_pred, target_names=class_names))

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('GAP Model - Confusion Matrix')
plt.tight_layout()
plt.savefig("custom_model_gap_cm.png")
plt.show()

model.save("custom_model_gap.h5")
print("\nCustom model with GAP head saved to custom_model_gap.h5")