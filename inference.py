#!/usr/bin/env python3
"""
Fire Detection Model Inference Script

This script loads a trained model and performs inference on test images,
generating a CSV file with predictions for Kaggle submission.

Usage:
    python inference.py --model_path models/best_model.keras --test_root datasets/test --output_csv submission.csv
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with linear warmup (for loading models trained with this schedule)."""

    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(tf.maximum(1, self.warmup_steps), tf.float32)
        total_steps = tf.cast(tf.maximum(1, self.total_steps), tf.float32)
        warmup_lr = self.base_lr * tf.minimum(1.0, step / warmup_steps)
        cosine_steps = tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.minimum(tf.maximum(step - warmup_steps, 0.0), cosine_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * progress / cosine_steps))
        cosine_lr = self.base_lr * cosine_decay
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

# Optional: focal loss alias for loading models saved with custom loss
def focal_loss(y_true, y_pred, gamma=1.5, alpha=0.75):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weights = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_mean(tf.reduce_sum(weights * cross_entropy, axis=1))

# Label mapping for Kaggle submission
LABEL_TO_ID = {
    'fire': 0,
    'no_fire': 1,
    'start_fire': 2
}

def load_model_and_classes(model_path, classes_path=None):
    """Load the trained model and class names."""
    # Load model (with custom_objects for focal loss if present)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "focal_loss": focal_loss,
            "WarmupCosineSchedule": WarmupCosineSchedule,
        }
    )
    print(f"Model loaded from {model_path}")

    # Load class names
    if classes_path:
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Try to find classes.txt in the same directory as the model
        model_dir = Path(model_path).parent
        classes_file = model_dir / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            print("Warning: classes.txt not found. Using default class names.")
            class_names = ['fire', 'no_fire', 'start_fire']

    print(f"Class names: {class_names}")
    return model, class_names


def get_target_size(model):
    """Return (height, width) from the model input shape."""
    input_shape = model.input_shape
    # Expected shape is (None, H, W, C)
    height, width = input_shape[1], input_shape[2]
    if height is None or width is None:
        raise ValueError(f"Model input shape is dynamic: {input_shape}")
    return int(height), int(width)

def preprocess_image(image_path, target_size):
    """Preprocess a single image for model input."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_single_image(model, image_path, class_names, target_size):
    """Predict class for a single image."""
    img_array = preprocess_image(image_path, target_size)
    predictions = model.predict(img_array, verbose=0)[0]

    # Get predicted class index and confidence
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_class = class_names[predicted_idx]

    return predicted_class, confidence, predictions

def collect_test_images(test_root):
    """Collect all test images and their paths."""
    test_images = []
    test_root = Path(test_root)

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    for ext in image_extensions:
        for img_path in test_root.rglob(f'*{ext}'):
            if img_path.is_file():
                test_images.append(img_path)

    # Sort by filename for consistent ordering
    test_images.sort(key=lambda x: x.name)

    print(f"Found {len(test_images)} test images")
    return test_images

def main():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--model_path', required=True, help='Path to trained model file')
    parser.add_argument('--test_root', required=True, help='Root directory of test images')
    parser.add_argument('--output_csv', default='submission.csv', help='Output CSV file path')
    parser.add_argument('--classes_file', help='Path to classes.txt file (optional)')

    args = parser.parse_args()

    # Load model and classes
    model, class_names = load_model_and_classes(args.model_path, args.classes_file)
    target_size = get_target_size(model)

    # Collect test images
    test_images = collect_test_images(args.test_root)

    if not test_images:
        print(f"No test images found in {args.test_root}")
        return

    # Perform inference
    results = []
    print("Running inference...")

    for i, img_path in enumerate(test_images):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_images)} images")

        try:
            # Get image ID (full filename with extension)
            image_id = img_path.name

            # Make prediction
            predicted_class, confidence, all_predictions = predict_single_image(
                model, str(img_path), class_names, target_size
            )

            results.append({
                'ID': image_id,
                'Label': LABEL_TO_ID[predicted_class]
            })

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Sort by ID for consistent ordering (try numeric sort if possible)
    try:
        # For IDs with extensions, sort by the numeric part before extension
        df['ID_numeric'] = df['ID'].str.extract(r'(\d+)').astype(float)
        df = df.sort_values('ID_numeric', na_position='last').drop('ID_numeric', axis=1).reset_index(drop=True)
    except:
        df = df.sort_values('ID').reset_index(drop=True)

    # Save to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print(f"Total predictions: {len(df)}")

    # Print summary
    if len(class_names) > 0:
        print("\nPrediction distribution:")
        print(df['Label'].value_counts())

if __name__ == '__main__':
    main()