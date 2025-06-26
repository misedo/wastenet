# run_final_custom_model_v2.py
# -------------------------------------------------------------
# End-to-end, fully-custom CNN for 10-class waste-image classification
# – fixes shuffle/label mismatch
# – adds proper input rescaling
# – upgrades architecture (2-conv blocks + GAP)
# – uses AdamW, label smoothing, early stopping, LR scheduler
# -------------------------------------------------------------
import os, time
import tensorflow as tf
import keras_cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     MaxPooling2D, GlobalAveragePooling2D,
                                     Dense, Dropout, Rescaling)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------------------------
# 0 - Environment sanity check
# ------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPUs found: {gpus}" if gpus else "❌ No GPU detected")
print(tf.__version__)

# ------------------------------------------------------------------
# 1 - Configuration
# ------------------------------------------------------------------
DATA_DIR   = '/workspace/project/wastedata'
IMG_HEIGHT = 224
IMG_WIDTH  = 224
BATCH_SIZE = 64
NUM_CLASSES= 10
EPOCHS     = 50
SEED       = 123

# ------------------------------------------------------------------
# 2 - KerasCV GPU-accelerated augmentation (train only)
# ------------------------------------------------------------------
augmenter = keras_cv.layers.RandomAugmentationPipeline(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal"),
        keras_cv.layers.RandomRotation(factor=0.1),
        keras_cv.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    augmentations_per_image=1,
)

# ------------------------------------------------------------------
# 3 - Robust tf.data loaders
#     * shuffle **True** for train
#     * shuffle **False** for val/test  → prevents label mis-alignment
# ------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    shuffle=False,                     # ← critical fix
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
print(f"Classes: {class_names}")

# ------------------------------------------------------------------
# 4 - Class-weight calculation (handles imbalance)
# ------------------------------------------------------------------
train_labels = np.concatenate([y for _, y in train_ds], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weights_dict = dict(enumerate(weights))
print(f"Class weights: {class_weights_dict}")

# ------------------------------------------------------------------
# 5 - Augment + prefetch pipeline
# ------------------------------------------------------------------
def apply_augmentations(images, labels):
    return augmenter(images, training=True), labels

train_ds = (train_ds
            .map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------
# 6 - Build CustomNetV2 (deeper, GAP head, still from scratch)
# ------------------------------------------------------------------
def conv_block(x, filters):
    for _ in range(2):
        x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
    return MaxPooling2D(2)(x)

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = Rescaling(1./255)(inputs)    # ← normalise once for all splits

for f in [32, 64, 128, 256]:
    x = conv_block(x, f)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

# ------------------------------------------------------------------
# 7 - Compile
# ------------------------------------------------------------------
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# ------------------------------------------------------------------
# 8 - Callbacks
# ------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# ------------------------------------------------------------------
# 9 - Train
# ------------------------------------------------------------------
start = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks,
)
print(f"Training time: {(time.time()-start)/60:.2f} min")

# ------------------------------------------------------------------
# 10 - Evaluate & sanity-check alignment
# ------------------------------------------------------------------
val_loss, val_acc = model.evaluate(val_ds, verbose=2)
print(f"Clean val accuracy: {val_acc:.3f}")

y_true = np.concatenate([y for _, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)
assert len(y_true) == len(y_pred), "Label/prediction count mismatch!"

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("CustomNetV2 – Confusion Matrix")
plt.tight_layout(); plt.savefig("custom_model_v2_cm.png"); plt.show()

# ------------------------------------------------------------------
# 11 - Save model & training curves
# ------------------------------------------------------------------
model.save("custom_model_v2.h5")
print("✅ Model saved to custom_model_v2.h5")

acc, val_acc_hist = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss_hist = history.history['loss'], history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Acc')
plt.plot(epochs_range, val_acc_hist, label='Val Acc')
plt.legend(); plt.title("Accuracy"); plt.xlabel("Epoch")

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss_hist, label='Val Loss')
plt.legend(); plt.title("Loss"); plt.xlabel("Epoch")

plt.tight_layout(); plt.savefig("custom_model_v2_history.png"); plt.show()
