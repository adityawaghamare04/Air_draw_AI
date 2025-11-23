# train_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# 🔹 Classes in your dataset and real_images/
labels = ['star']
num_classes = len(labels)

DATASET_DIR = "dataset"

def load_quickdraw_data():
    X_list = []
    y_list = []

    for idx, label in enumerate(labels):
        path = os.path.join(DATASET_DIR, f"{label}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")

        data = np.load(path)  # shape (N, 28, 28) or (N, 784)
        print(f"{label} dataset shape:", data.shape)

        if data.ndim == 2:      # (N, 784)
            data = data.reshape(-1, 28, 28)
        elif data.ndim == 3:    # (N, 28, 28)
            pass
        else:
            raise ValueError(f"Unexpected shape for {label}: {data.shape}")

        X_list.append(data)
        y_list.append(np.full((data.shape[0],), idx, dtype=np.int32))

    X = np.concatenate(X_list, axis=0)   # (Total_N, 28, 28)
    y = np.concatenate(y_list, axis=0)   # (Total_N,)

    # Normalize + add channel dimension
    X = X.astype("float32") / 255.0
    X = X[..., np.newaxis]              # (N, 28, 28, 1)

    y_cat = to_categorical(y, num_classes=num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_val, y_train, y_val

def build_model(input_shape=(28, 28, 1), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_quickdraw_data()

    model = build_model(num_classes=len(labels))

    history = model.fit(
        X_train, y_train,
        epochs=10,          # increase if you want better accuracy
        batch_size=64,
        validation_data=(X_val, y_val)
    )

    model.save("airdraw_model.keras", include_optimizer=False)
    print("✅ Trained model saved as airdraw_model.keras")
