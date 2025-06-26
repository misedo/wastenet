# train_transfer_learning.py
import os
import tensorflow as tf

# --- This script uses the recommended method for suppressing logs ---
# It should be run from the command line like this:
# TF_CPP_MIN_LOG_LEVEL='2' python train_transfer_learning.py

# --- In-script check to confirm GPU detection ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! TensorFlow has found and is using the following GPU(s): {gpus}")
else:
    print("❌ Error! TensorFlow did NOT find any GPUs.")


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Import the specific preprocessing function for MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
DATA_DIR = '/workspace/project/data/dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64  # You can try increasing this to 64 for your powerful GPU
NUM_CLASSES = 6
EPOCHS = 30 # Transfer learning converges much faster

# --- 1. Data Preparation with MobileNetV2 Preprocessing ---
print("Preparing Data Generators with MobileNetV2 preprocessing...")

# Note: We use the dedicated 'preprocess_input' function instead of 'rescale=1./255'
# Each pre-trained model has its own specific way of normalizing pixels.
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Use the correct preprocessing
    validation_split=0.2,
    rotation_range=20,
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
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# --- 2. Model Definition (Transfer Learning) ---
print("\nBuilding model with MobileNetV2 base...")

# Load MobileNetV2 pre-trained on ImageNet, without its final classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the convolutional base. This is the key to transfer learning.
# We don't want to re-train the parts that have already learned about edges, textures, etc.
base_model.trainable = False

# Create our new custom classifier "head" to attach to the base
x = base_model.output
x = GlobalAveragePooling2D()(x) # A good alternative to Flatten()
x = Dense(128, activation='relu')(x) # A dense layer to learn combinations of features
x = Dropout(0.5)(x) # Regularization
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Our final output layer

# Combine the frozen base and our new head into the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
# A slightly higher learning rate like 0.001 is fine here because we are only training the small new head.
model.compile(
    optimizer=Adam(learning_rate=0.001),
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
    # Use parallel workers to prevent CPU bottleneck and keep the GPU fed
    workers=8,
    use_multiprocessing=True
)

# --- 4. Model Evaluation (Same as before for direct comparison) ---
print("\nEvaluating Model...")

# Plotting training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc)) # Use len(acc) to be safe

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
plt.savefig("transfer_learning_training_history.png") # Save the plot
plt.show()

# Classification Report and Confusion Matrix
print("\nGenerating Classification Report and Confusion Matrix...")
# Ensure we get predictions for the entire validation set
Y_pred = model.predict(validation_generator, steps=np.ceil(validation_generator.samples / BATCH_SIZE))
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

print('Classification Report')
print(classification_report(y_true, y_pred, target_names=list(train_generator.class_indices.keys())))

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(train_generator.class_indices.keys()),
            yticklabels=list(train_generator.class_indices.keys()))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig("transfer_learning_confusion_matrix.png") # Save the plot
plt.show()

# Save the trained model
model.save("transfer_learning_model.h5")
print("\nTransfer learning model saved to transfer_learning_model.h5")