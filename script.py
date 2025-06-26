# train_wastenet.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! TensorFlow has found and is using the following GPU(s): {gpus}")
else:
    print("❌ Error! TensorFlow did NOT find any GPUs.")

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
DATA_DIR = '/workspace/project/wastedata'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 30

# --- 1. Data Preparation and Augmentation ---
print("Preparing Data Generators...")

# Use ImageDataGenerator for loading and augmenting data
datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0, 1]
    validation_split=0.2,        # Split data into 80% training, 20% validation
    rotation_range=20,           # Augmentation settings
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Set as validation data
)

# --- 2. Model Definition (WasteNet) ---
print("\nBuilding WasteNet Model...")

model = Sequential([
    # Input Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Flatten and Classify
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Regularization to prevent overfitting
    Dense(NUM_CLASSES, activation='softmax') # Output layer for 10 classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 3. Model Training ---
print("\nStarting Training...")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    workers=4,  # Use multiple workers for data loading
    use_multiprocessing=True  # Use multiprocessing for data loading
)

# --- 4. Model Evaluation ---
print("\nEvaluating Model...")

# Plotting training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Classification Report and Confusion Matrix
# Note: For a true test set evaluation, you'd create a third generator
# Here we use the validation set for demonstration.
print("\nGenerating Classification Report and Confusion Matrix...")
Y_pred = model.predict(validation_generator, validation_generator.samples // BATCH_SIZE + 1)
y_pred = np.argmax(Y_pred, axis=1)

print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=list(train_generator.class_indices.keys())))

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the trained model
model.save("wastenet_model.h5")
print("\nWasteNet model saved to wastenet_model.h5")