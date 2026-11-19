import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

IMAGE_SIZE = (64, 64)
SEED = 42

def find_image_root(data_dir: Path) -> Path:
    candidates = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    for folder in candidates:
        class_dirs = [p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()] if folder.exists() else []
        ids = {int(p.name) for p in class_dirs}
        if len(ids) >= 2:
            return folder
    raise FileNotFoundError(
        f"Could not find numeric class folders under {data_dir}. "
        "Extract BanglaLekha-Isolated so that Images/1 ... Images/84 exist."
    )

def load_data(root: Path, max_per_class=None):
    images, labels = [], []
    class_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )
    if not class_dirs:
        raise RuntimeError(f"No numeric class directories found in {root}")
    class_ids = [int(p.name) for p in class_dirs]
    if class_ids != list(range(1, 85)):
        print(f"Warning: found {len(class_ids)} classes; expected folders 1..84.")
    for class_dir in class_dirs:
        files = [p for p in class_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        files.sort()
        if max_per_class:
            files = files[:max_per_class]
        for file in files:
            try:
                img = tf.keras.utils.load_img(file, color_mode="grayscale", target_size=IMAGE_SIZE)
                arr = tf.keras.utils.img_to_array(img) / 255.0
                images.append(arr)
                labels.append(int(class_dir.name) - 1)
            except Exception as exc:
                print(f"Skipping {file}: {exc}")
    if not images:
        raise RuntimeError("No images loaded. Check the dataset folder structure.")
    return np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32), class_ids

def build_model(num_classes, dropout=0.35):
    inputs = tf.keras.Input(shape=(64, 64, 1))
    x = tf.keras.layers.RandomRotation(0.04)(inputs)
    x = tf.keras.layers.RandomTranslation(0.06, 0.06)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout / 2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def run_experiment(name, x_train, y_train, x_val, y_val, args, lr, dropout):
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "image_size": "64x64", "channels": 1, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": lr,
            "dropout": dropout, "optimizer": "Adam",
            "architecture": "3-block CNN", "num_classes": int(len(np.unique(y_train))),
            "dataset": "BanglaLekha-Isolated"
        })
        model = build_model(int(max(y_train.max(), y_val.max()) + 1), dropout)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, mode="max", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ]
        history = model.fit(
            x_train, y_train, validation_data=(x_val, y_val),
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=callbacks, verbose=1
        )
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
        mlflow.log_metrics({
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc)
        })
        os.makedirs("models", exist_ok=True)
        model.save("models/model.keras")
        mlflow.log_artifact("models/model.keras")
        return float(val_acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional quick-run cap per class")
    parser.add_argument("--tracking-uri", default="file:./artifacts/mlflow")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    root = find_image_root(Path(args.data_dir))
    print(f"Loading dataset from: {root}")
    x, y, class_ids = load_data(root, args.max_per_class)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # Dataset folder IDs are 1..84; the bundled mapping uses zero-based model indices.
    with open("labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    labels = {str(i): labels[str(i)] for i in range(len(class_ids))}
    with open("labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Bangla-Handwritten-Character-Recognition")
    # Two separately logged runs are created in every invocation.
    run_experiment("baseline_lr_1e-3", x_train, y_train, x_val, y_val, args, 1e-3, 0.35)
    run_experiment("regularized_lr_3e-4", x_train, y_train, x_val, y_val, args, 3e-4, 0.50)
    print("Training complete. Best model from the second run is saved at models/model.keras")
    print("Start MLflow: mlflow ui --backend-store-uri file:./artifacts/mlflow --host 0.0.0.0 --port 5000")

if __name__ == "__main__":
    main()
