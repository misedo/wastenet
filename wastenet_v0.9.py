# run_custom_model_final_with_all_outputs.py
import os
import time
import datetime
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import matplotlib.cm as cm
import matplotlib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

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
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
NUM_CLASSES = 10
EPOCHS = 10

# --- 2. Data Pipeline ---
print("Preparing tf.data.Dataset pipeline for Custom Model...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DATA_DIR, label_mode='int', seed=123, shuffle=True,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DATA_DIR, label_mode='int', seed=123, shuffle=False,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR, label_mode='int', seed=123, shuffle=False,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE)
class_names = train_ds.class_names
print(f"Found Classes: {class_names}")

unprocessed_test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DATA_DIR, label_mode='int', seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Augmentation & Preprocessing ---
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode="horizontal"),
    tf.keras.layers.RandomRotation(factor=0.15),
    tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
])
def apply_augmentations(images, labels): return augmenter(images, training=True), labels
train_ds = train_ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- 4. Class Weights ---
train_labels = np.concatenate([y for x, y in train_ds], axis=0)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(weights))

# --- 5. Build the Custom CNN Model ---
print("\nBuilding CustomNetV2 Model...")
inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = Rescaling(1./255)(inputs)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), padding='same', activation='relu', name='last_conv_layer')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- 6. Train the Model ---
log_dir = "logs/custom_model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
print("\n--- Starting Training for Custom Model ---")
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    class_weight=class_weights_dict, callbacks=[early_stopping_callback, tensorboard_callback]
)

# --- 7. Evaluate & Plot on Validation Data ---
print("\n--- Generating Plots based on VALIDATION Data for Custom Model---")
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
print('\nValidation Set Classification Report')
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_true, y_pred, target_names=class_names))

save_report_as_image(report_dict, 
                     title='Custom Model - Validation Metrics', 
                     filename='custom_model_validation_report_table.png')

confusion_mat = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Custom Model: Validation Set Confusion Matrix')
plt.savefig("custom_model_validation_cm.png")
plt.close()

# --- 8. Final Evaluation on Unseen TEST Data ---
print("\n--- Evaluating Final Custom Model on Unseen TEST Data ---")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\n# Final Test Data - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# --- 9. Generate and Save Grad-CAM Visualizations ---
print("\n--- Generating Grad-CAM Visualizations for Custom Model ---")
last_conv_layer_name = "last_conv_layer"
gradcam_output_dir = "gradcam_visualizations_custom"
os.makedirs(gradcam_output_dir, exist_ok=True)
original_images, original_labels = next(iter(unprocessed_test_ds))
processed_images_for_pred, _ = next(iter(test_ds))
num_images_to_visualize = 10

for i in range(num_images_to_visualize):
    original_img = original_images[i].numpy()
    img_array_for_pred = np.expand_dims(processed_images_for_pred[i], axis=0)
    preds = model.predict(img_array_for_pred)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    true_class_name = class_names[original_labels[i]]
    print(f"Image {i+1}: True Label='{true_class_name}', Predicted='{predicted_class_name}'")
    heatmap = make_gradcam_heatmap(img_array_for_pred, model, last_conv_layer_name)
    filename = f"gradcam_custom_img{i+1}_true-{true_class_name}_pred-{predicted_class_name}.jpg"
    cam_path = os.path.join(gradcam_output_dir, filename)
    save_gradcam(original_img, heatmap, cam_path)

print(f"\n{num_images_to_visualize} Grad-CAM images saved in '{gradcam_output_dir}' directory.")

# --- 10. Save Final Model ---
model.save("custom_model_final.keras")
print("\nCustom model saved to custom_model_final.keras")