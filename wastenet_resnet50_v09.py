# run_resnet50_final_with_all_outputs.py
import os
import time
import datetime
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2 
import matplotlib.cm as cm
import matplotlib
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

# --- HELPER FUNCTIONS (MOVED TO TOP) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.6):
    img = img.astype(np.uint8)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

def save_report_as_image(report, title='Classification Report', filename='report_table.png'):
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)
    for col in ['precision', 'recall', 'f1-score']:
        if col in df: df[col] = df[col].round(3)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off'); ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Classification report table saved to {filename}")

# --- 1. Configuration ---
TRAIN_DATA_DIR = 'wastedata_split/train'
VALIDATION_DATA_DIR = 'wastedata_split/val'
TEST_DATA_DIR = 'wastedata_split/test' 
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
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR, label_mode='int', seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False)
class_names = train_ds.class_names
print(f"Found Classes: {class_names}")
unprocessed_test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR, label_mode='int', seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False)

def resnet_preprocess(images, labels):
    return preprocess_input(images), labels

train_ds = train_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.map(resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 3. Build ResNet50 Model ---
print("\nBuilding ResNet50 Transfer Learning Model...")
base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. Callbacks Setup ---
log_dir = "logs/resnet50_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

# --- 5. Two-Phase Training ---
print("\n--- Phase 1: Training Head (Freezing Base Model) ---")
base_model.trainable = False
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=INITIAL_EPOCHS, validation_data=val_ds, callbacks=[tensorboard_callback])
print("\n--- Phase 2: Fine-Tuning (Unfreezing Top Layers) ---")
base_model.trainable = True
fine_tune_at = 143
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_ds, epochs=TOTAL_EPOCHS, initial_epoch=history.epoch[-1], validation_data=val_ds, callbacks=[tensorboard_callback, early_stopping_callback])

# --- 6. Evaluate & Plot on Validation Data ---
print("\n--- Generating Plots based on VALIDATION Data for ResNet50 Model ---")
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(x=INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(x=INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.suptitle('ResNet50 Model Training History', fontsize=16)
plt.savefig("resnet50_model_training_history.png")
plt.close()

y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
print('\nValidation Set Classification Report')
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_names))
save_report_as_image(report_dict, 
                     title='ResNet50 Model - Validation Metrics', 
                     filename='resnet50_model_validation_report_table.png')

confusion_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('ResNet50 Model: Validation Set Confusion Matrix')
plt.savefig("resnet50_model_validation_cm.png")
plt.close()

# --- 7. Final Evaluation on Unseen TEST Data ---
print("\n--- Evaluating Final ResNet50 Model on Unseen TEST Data ---")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\n#####################################################")
print(f"# Final Test Data - Loss:     {test_loss:.4f}")
print(f"# Final Test Data - Accuracy: {test_accuracy:.4f}")
print(f"#####################################################")

# --- 8. Generate and Save Grad-CAM Visualizations ---
print("\n--- Generating Grad-CAM Visualizations for ResNet50 Model ---")
last_conv_layer_name = "conv5_block3_out"
gradcam_output_dir = "gradcam_visualizations_resnet"
os.makedirs(gradcam_output_dir, exist_ok=True)
original_images, original_labels = next(iter(unprocessed_test_ds))
processed_images, _ = next(iter(test_ds))
num_images_to_visualize = 10

for i in range(num_images_to_visualize):
    original_img = original_images[i].numpy()
    img_array = np.expand_dims(processed_images[i], axis=0)
    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    true_class_name = class_names[original_labels[i]]
    print(f"Image {i+1}: True Label='{true_class_name}', Predicted='{predicted_class_name}'")
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    filename = f"gradcam_resnet_img{i+1}_true-{true_class_name}_pred-{predicted_class_name}.jpg"
    cam_path = os.path.join(gradcam_output_dir, filename)
    save_gradcam(original_img, heatmap, cam_path)

print(f"\n{num_images_to_visualize} Grad-CAM images saved in '{gradcam_output_dir}' directory.")

# --- 9. Save Final Model ---
model.save("resnet50_model_final.keras")
print("\nResNet50 model saved to resnet50_model_final.keras")