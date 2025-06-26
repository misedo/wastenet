# train_and_compare_all.py
import os
import time
import tensorflow as tf
import keras_cv

# --- This script will run all three models and print a final summary ---
# It should be run like this to suppress logs:
# TF_CPP_MIN_LOG_LEVEL='2' python train_and_compare_all.py

# --- In-script check to confirm GPU detection ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Success! TensorFlow has found and is using the following GPU(s): {gpus}")
else:
    print("❌ Error! TensorFlow did NOT find any GPUs.")


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn.utils import class_weight

# --- Configuration ---
# IMPORTANT: Update this path to your new 10-class dataset
DATA_DIR = '/workspace/project/wastedata'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 64
NUM_CLASSES = 10
INITIAL_EPOCHS = 10  # Epochs for feature extraction
FINETUNE_EPOCHS = 20   # Epochs for fine-tuning
TOTAL_EPOCHS = INITIAL_EPOCHS + FINETUNE_EPOCHS


def get_model_and_preprocessing(model_name):
    """Returns the base model class, its preprocessing function, and fine-tune layer index."""
    if model_name == 'ResNet50':
        from tensorflow.keras.applications import ResNet50 as BaseModel
        from tensorflow.keras.applications.resnet import preprocess_input
        fine_tune_at = 143
        return BaseModel, preprocess_input, fine_tune_at
    elif model_name == 'MobileNetV2':
        from tensorflow.keras.applications import MobileNetV2 as BaseModel
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        fine_tune_at = 100
        return BaseModel, preprocess_input, fine_tune_at
    elif model_name == 'EfficientNetB0':
        from tensorflow.keras.applications import EfficientNetB0 as BaseModel
        from tensorflow.keras.applications.efficientnet import preprocess_input
        fine_tune_at = 180
        return BaseModel, preprocess_input, fine_tune_at
    else:
        raise ValueError("Invalid model_name.")

# --- Main Experiment ---
models_to_compare = ['ResNet50', 'MobileNetV2', 'EfficientNetB0']
all_results = {}

for model_name in models_to_compare:
    print("\n" + "="*80)
    print(f"    STARTING EXPERIMENT FOR MODEL: {model_name}")
    print("="*80 + "\n")

    # --- 1. Get Model-Specific Components ---
    BaseModel, preprocess_input, fine_tune_at = get_model_and_preprocessing(model_name)

    # --- 2. Data Pipeline ---
    print("Preparing tf.data.Dataset pipeline...")
    # Create dataset objects first to calculate class weights
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    
    # --- NEW: Calculate Class Weights to handle imbalance ---
    # This iterates through the dataset once to get all labels
    print("Calculating class weights to handle data imbalance...")
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Calculated Class Weights:\n{class_weights_dict}\n")
    # --- End of New Section ---

    # --- 3. KerasCV Augmentation and Final Pipeline ---
    augmenter = keras_cv.layers.RandomAugmentationPipeline(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal"),
            keras_cv.layers.RandomRotation(factor=0.1),
        ],
        augmentations_per_image=1,
    )

    def apply_augmentations(images, labels):
        return augmenter(images, training=True), labels

    train_ds = train_ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
    
    def preprocess_data(images, labels):
        return preprocess_input(images), labels
        
    train_ds = train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- 4. Model Building and Training ---
    print(f"Building model with {model_name} base...")
    base_model = BaseModel(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Start timer
    start_time = time.time()

    # Phase 1: Feature Extraction
    base_model.trainable = False
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"\n--- Training head for {model_name} ---")
    history = model.fit(
        train_ds,
        epochs=INITIAL_EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights_dict # APPLY THE WEIGHTS HERE
    )

    # Phase 2: Fine-Tuning
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(f"\n--- Fine-tuning {model_name} ---")
    history_fine = model.fit(
        train_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        class_weight=class_weights_dict # AND APPLY THEM HERE TOO
    )
    
    # End timer
    end_time = time.time()
    training_duration_seconds = end_time - start_time

    # --- 5. Store Results ---
    final_val_accuracy = history_fine.history['val_accuracy'][-1]
    final_val_loss = history_fine.history['val_loss'][-1]
    
    all_results[model_name] = {
        'val_accuracy': final_val_accuracy,
        'val_loss': final_val_loss,
        'training_time_seconds': training_duration_seconds
    }
    
    model.save(f"{model_name}_final_model.h5")
    print(f"Saved final model to {model_name}_final_model.h5")


# --- 6. Final Comparison Summary ---
print("\n" + "="*80)
print("    ✅✅✅ FINAL EXPERIMENT SUMMARY ✅✅✅")
print("="*80 + "\n")

print(f"{'Model':<20} | {'Validation Accuracy':<22} | {'Validation Loss':<18} | {'Training Time (min)':<20}")
print("-"*85)

for model_name, metrics in all_results.items():
    acc_str = f"{metrics['val_accuracy']:.4f}"
    loss_str = f"{metrics['val_loss']:.4f}"
    time_str = f"{metrics['training_time_seconds']/60:.2f}"
    print(f"{model_name:<20} | {acc_str:<22} | {loss_str:<18} | {time_str:<20}")

# Determine and announce the winner
if all_results:
    best_model_name = max(all_results, key=lambda name: all_results[name]['val_accuracy'])
    print("\n" + "-"*85)
    print(f"🏆 Best performing model based on Validation Accuracy: {best_model_name}")
    print("-"*85)
else:
    print("No models were trained to determine a winner.")