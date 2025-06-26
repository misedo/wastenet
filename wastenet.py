# run_final_custom_model.py
import os
import time
import datetime
import io

import tensorflow as tf
import keras_cv

# --- Suppress TensorFlow logs ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# --- 1. Configuration ---
# IMPORTANT: These paths should point to the folders created by the split_data.py script
TRAIN_DATA_DIR = '/workspace/project/wastedata_split/train'
VALIDATION_DATA_DIR = '/workspace/project/wastedata_split/val'

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
NUM_CLASSES = 10
EPOCHS = 10

# --- 2. KerasCV Augmentation Pipeline (GPU Accelerated) ---
print("Defining KerasCV GPU-accelerated augmentation pipeline...")
augmenter = keras_cv.layers.RandomAugmentationPipeline(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal"),
        keras_cv.layers.RandomRotation(factor=0.15),
        keras_cv.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    augmentations_per_image=1,
)

# --- 3. High-Performance Data Loading with Explicit Directories ---
print("Preparing tf.data.Dataset pipeline from explicit train/val folders...")

# Create the training dataset from the 'train' folder
# Note: We remove validation_split and subset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR,
    label_mode='int', # Default, produces integer labels for sparse_categorical_crossentropy
    seed=123,
    shuffle=True, # Shuffle the dataset for better training
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Create the validation dataset from the 'val' folder
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DATA_DIR,
    label_mode='int',
    seed=123,
    shuffle=False, # No need to shuffle validation data
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Found classes: {class_names}")

# --- 4. Calculate Class Weights to Handle Imbalance ---
print("\nCalculating class weights from the training set...")
train_labels = np.concatenate([y for x, y in train_ds], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(weights))
print(f"Calculated Class Weights:\n{class_weights_dict}\n")

# --- 5. Apply Augmentations and Prefetch for Performance ---
def apply_augmentations(images, labels):
    return augmenter(images, training=True), labels

train_ds = train_ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 6. Build the Custom CNN Model ("CustomNetV2") ---
print("Building CustomNetV2 Model...")
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = Rescaling(1./255)(inputs)
# ... (rest of model architecture is the same) ...
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
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# --- 7. Compile and Visualize the Model ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

try:
    plot_model(model, to_file='custom_model_final_architecture.png', show_shapes=True, show_layer_activations=True)
    print("\n✅ Model architecture plot saved to custom_model_final_architecture.png\n")
except Exception as e:
    print(f"\n❌ Could not create plot. Ensure graphviz and pydot are installed. Error: {e}")

model.summary()

# --- 8. Train the Model ---
# Define the EarlyStopping callback to prevent overfitting
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

print("\nStarting Training...")
start_time = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[early_stopping_callback]
)
end_time = time.time()
print(f"\nTotal training time: {(end_time - start_time)/60:.2f} minutes")

# --- 9. Evaluate the Model & Save Artifacts ---
print("\nEvaluating Model...")

# Plotting training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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
model.save("custom_model_final.h5")
print("\nCustom model saved to custom_model_final.h5")