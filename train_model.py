"""
Improved training script for digit recognizer.

- Uses MNIST (0-9) by default (10 classes).
- Adds stronger augmentation to mimic photographed digits.
- Adds callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).
- Saves best model to digit_model_best.h5 and final digit_model.h5.
- Tries to produce a TFLite export.
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 64
EPOCHS = 30
MODEL_PATH = "digit_model.h5"
BEST_MODEL_PATH = "digit_model_best.h5"

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

def preprocessing_fn(img):
    # img is float32 in [0,1]
    if np.random.rand() < 0.5:
        factor = 0.9 + 0.2 * np.random.rand()
        img = np.clip(img * factor, 0.0, 1.0)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0.0, 1.0)
    if np.random.rand() < 0.2:
        import cv2
        arr = (img[..., 0] * 255).astype("uint8")
        if np.random.rand() < 0.5:
            k = np.ones((2, 2), np.uint8)
            arr = cv2.dilate(arr, k, iterations=1)
        else:
            k = np.ones((2, 2), np.uint8)
            arr = cv2.erode(arr, k, iterations=1)
        arr = (arr.astype("float32") / 255.0).reshape(img.shape)
        img = arr
    return img

datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
    shear_range=5,
    preprocessing_function=preprocessing_fn,
    fill_mode='nearest'
)
datagen.fit(x_train)

print("Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.15),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print("Training...")
history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE, seed=SEED),
    epochs=EPOCHS,
    validation_data=(x_test, y_test_cat),
    callbacks=callbacks,
    verbose=2
)

loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test accuracy: {acc:.4f}  loss: {loss:.4f}")

model.save(MODEL_PATH)
print(f"Saved final model to {MODEL_PATH}")
if os.path.exists(BEST_MODEL_PATH):
    print(f"Best model saved to {BEST_MODEL_PATH}")

# Optional: try TFLite conversion
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open('digit_model.tflite', 'wb').write(tflite_model)
    print("Saved quantized TFLite model to digit_model.tflite")
except Exception as e:
    print("TFLite conversion skipped or failed:", e)
