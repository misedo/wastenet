# train_custom_model_v2.py
import os
import tensorflow as tf

# --- Suppress TensorFlow logs. Run script from shell like this: ---
# TF_CPP_MIN_LOG_LEVEL='2' python train_custom_model_v2.py

# --- In-script check to confirm GPU detection ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! TensorFlow has found and is using the following GPU(s): {gpus}")
else:
    print("❌ Error! TensorFlow did NOT find any GPUs.")


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model # For the nice visualization

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# --- Configuration ---
# IMPORTANT: Update this path to your new dataset's location inside the container
DATA_DIR = '/workspace/project/wastedata'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
NUM_CLASSES = 10 # IMPORTANT: Updated for the new dataset
EPOCHS = 50      # A deeper custom model needs more epochs to train

# --- 1. Data Preparation and Augmentation ---
print("Preparing Data Generators...")

# For a custom model, we use simple pixel rescaling.
# The 'preprocess_input' function is specific to pre-trained models.
datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0, 1]
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

# --- 2. Build a Better Custom Model ("CustomNetV2") ---
print("\nBuilding CustomNetV2 Model...")

# Define the input layer using modern Keras syntax
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Block 1
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

# Block 2
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

# Block 3
x = Conv2D(128, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

# Block 4
x = Conv2D(256, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)

# Classifier Head
x = Flatten()(x)
x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the final model object
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Generate the "Nicer" Model Visualization ---
try:
    plot_model(model, to_file='custom_model_v2_architecture.png', show_shapes=True, show_layer_activations=True)
    print("\n✅ Model architecture plot saved to custom_model_v2_architecture.png\n")
except ImportError as e:
    print(f"\n❌ Could not create plot. Please ensure graphviz and pydot are installed. Error: {e}")

model.summary()


# --- 3. Model Training ---
print("\nStarting Training...")

# Calculate class weights to handle potential class imbalance in the new dataset
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"\nUsing Class Weights: {class_weights_dict}\n")


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights_dict, # Use the calculated weights
    workers=8,
    use_multiprocessing=True
)

# --- 4. Model Evaluation ---
print("\nEvaluating Model...")

# Plotting training history
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
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("custom_model_v2_training_history.png") # Save the plot
plt.show()

# Classification Report and Confusion Matrix
print("\nGenerating Classification Report and Confusion Matrix...")

# Get true labels from the validation generator
y_true = validation_generator.classes
# Predict probabilities on the validation set
Y_pred_probs = model.predict(validation_generator, steps=np.ceil(validation_generator.samples / BATCH_SIZE))
# Convert probabilities to class predictions
y_pred = np.argmax(Y_pred_probs, axis=1)

# Ensure the number of predictions matches the number of true labels
if len(y_pred) != len(y_true):
    y_true = y_true[:len(y_pred)]

print('\nClassification Report')
# Get class names from the generator
class_names = list(train_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_names))

print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig("custom_model_v2_confusion_matrix.png") # Save the plot
plt.show()

# Save the trained model
model.save("custom_model_v2.h5")
print("\nCustom model v2 saved to custom_model_v2.h5")