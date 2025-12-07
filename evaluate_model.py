#!/usr/bin/env python3
"""
Fire Detection Model Evaluation Script

This script evaluates a trained model on validation/test data and generates
performance metrics, confusion matrix, and classification report.

Usage:
    python evaluate_model.py --model_path models/best_model.keras --test_root datasets/test --output_dir evaluation/
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from PIL import Image

# Constants
IMG_HEIGHT = 299  # InceptionV3 input size
IMG_WIDTH = 299

def load_model_and_classes(model_path, classes_path=None):
    """Load the trained model and class names."""
    # Load model
    model = tf.keras.models.load_model(model_path)
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

def preprocess_image(image_path):
    """Preprocess a single image for model input."""
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to array and normalize
    img_array = np.array(img) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_single_image(model, image_path):
    """Predict class probabilities for a single image."""
    # Preprocess image
    img_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)[0]

    return predictions

def collect_test_images_with_labels(test_root):
    """Collect test images and their true labels from directory structure."""
    test_data = []
    test_root = Path(test_root)

    # Find all class directories
    class_dirs = [d for d in test_root.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        for ext in image_extensions:
            for img_path in class_dir.rglob(f'*{ext}'):
                if img_path.is_file():
                    test_data.append({
                        'filepath': str(img_path),
                        'true_class': class_name
                    })

    # Sort by filepath for consistent ordering
    test_data.sort(key=lambda x: x['filepath'])

    print(f"Found {len(test_data)} test images with labels")
    return test_data

def evaluate_model(model, test_data, class_names):
    """Evaluate model on test data."""
    y_true = []
    y_pred = []
    y_prob = []

    print("Running evaluation...")

    for i, item in enumerate(test_data):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_data)} images")

        filepath = item['filepath']
        true_class = item['true_class']

        try:
            # Get predictions
            predictions = predict_single_image(model, filepath)

            # Get predicted class
            predicted_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_idx]

            # Store results
            y_true.append(true_class)
            y_pred.append(predicted_class)
            y_prob.append(predictions)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    return y_true, y_pred, np.array(y_prob)

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_probabilities(y_prob, y_true, class_names, output_path):
    """Plot probability distributions for each class."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))

    for i, class_name in enumerate(class_names):
        # Get probabilities for this class
        class_probs = y_prob[:, i]

        # Separate by true class
        true_class_mask = np.array(y_true) == class_name

        axes[i].hist(class_probs[~true_class_mask], alpha=0.5, label='Other classes', bins=20)
        axes[i].hist(class_probs[true_class_mask], alpha=0.5, label=f'True {class_name}', bins=20)
        axes[i].set_xlabel('Predicted Probability')
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'Probability Distribution for {class_name}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_report(y_true, y_pred, class_names, output_path):
    """Save detailed metrics report."""
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average=None
    )

    # Macro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average='macro'
    )

    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, average='weighted'
    )

    # Create report
    report = f"""
Fire Detection Model Evaluation Report
=====================================

Overall Metrics:
- Accuracy: {accuracy:.4f}

Macro Averages:
- Precision: {macro_precision:.4f}
- Recall: {macro_recall:.4f}
- F1-Score: {macro_f1:.4f}

Weighted Averages:
- Precision: {weighted_precision:.4f}
- Recall: {weighted_recall:.4f}
- F1-Score: {weighted_f1:.4f}

Per-Class Metrics:
"""

    for i, class_name in enumerate(class_names):
        report += f"""
{class_name}:
  - Precision: {precision[i]:.4f}
  - Recall: {recall[i]:.4f}
  - F1-Score: {f1[i]:.4f}
  - Support: {support[i]}
"""

    # Classification report
    report += f"""

Detailed Classification Report:
{classification_report(y_true, y_pred, labels=class_names, target_names=class_names)}
"""

    # Save to file
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Metrics report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', required=True, help='Path to trained model file')
    parser.add_argument('--test_root', required=True, help='Root directory of test images with labels')
    parser.add_argument('--output_dir', default='evaluation', help='Output directory for results')
    parser.add_argument('--classes_file', help='Path to classes.txt file (optional)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and classes
    model, class_names = load_model_and_classes(args.model_path, args.classes_file)

    # Collect test data with labels
    test_data = collect_test_images_with_labels(args.test_root)

    if not test_data:
        print(f"No test images found in {args.test_root}")
        return

    # Evaluate model
    y_true, y_pred, y_prob = evaluate_model(model, test_data, class_names)

    # Save predictions
    predictions_df = pd.DataFrame({
        'true_class': y_true,
        'predicted_class': y_pred
    })
    for i, class_name in enumerate(class_names):
        predictions_df[f'{class_name}_prob'] = y_prob[:, i]

    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # Generate plots
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    prob_path = output_dir / 'class_probabilities.png'
    plot_class_probabilities(y_prob, y_true, class_names, prob_path)
    print(f"Probability distributions saved to {prob_path}")

    # Save metrics report
    report_path = output_dir / 'evaluation_report.txt'
    save_metrics_report(y_true, y_pred, class_names, report_path)

    print("Evaluation completed!")

if __name__ == '__main__':
    main()
