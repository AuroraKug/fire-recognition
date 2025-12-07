"""Utilities for scanning fire-detection datasets and preparing training data.

This module centralises the dataset handling so we can effortlessly mix the
three official training archives, apply deterministic splits, and build Tensor
Flow ``tf.data.Dataset`` objects for training and inference. It is designed to
work with the directory layout::::

datasets/
    train/
        FIRE_DATABASE_1/
        FIRE_DATABASE_2/
        FIRE_DATABASE_3/
    val/
    test/

Each database is expected to contain class sub-directories (``fire`` or ``Feu``,
``no_fire`` or ``Pas de Feu``...). The helper functions normalise these names so
that the rest of the pipeline can operate with the canonical label ids:

    0 -> fire
    1 -> no fire
    2 -> start fire

Example
-------
>>> from dataset import build_image_index, stratified_split, make_tf_dataset
>>> df = build_image_index(["datasets/train/FIRE_DATABASE_1"], split="train")
>>> train_df, val_df = stratified_split(df, val_fraction=0.1, seed=42)
>>> train_ds = make_tf_dataset(train_df, batch_size=32, image_size=(224, 224))

The resulting dataframe can be saved to CSV for auditing or fed directly into
model training utilities.
"""

from __future__ import annotations

import math
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import tensorflow as tf


# Canonical label mapping ---------------------------------------------------

CANONICAL_LABELS = ["fire", "no_fire", "start_fire"]
CANONICAL_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_LABELS)}
ID_TO_CANONICAL = {idx: name for name, idx in CANONICAL_TO_ID.items()}

CLASS_ALIASES = {
    "fire": "fire",
    "feu": "fire",
    "0": "fire",
    "no fire": "no_fire",
    "no_fire": "no_fire",
    "pas de feu": "no_fire",
    "1": "no_fire",
    "start fire": "start_fire",
    "start_fire": "start_fire",
    "debut de feu": "start_fire",
    "debut feu": "start_fire",
    "debut": "start_fire",
    "debut_of_fire": "start_fire",
    "2": "start_fire",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# Data structures -----------------------------------------------------------


@dataclass
class ImageRecord:
    """Small container describing a single image sample."""

    path: Path
    label_id: int
    label_name: str
    class_dir: str
    split: str


# Helpers ------------------------------------------------------------------


def _normalise_string(value: str) -> str:
    """Return a lowercase ASCII-only string for comparison purposes."""

    normalised = unicodedata.normalize("NFKD", value)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.replace("-", " ").replace("_", " ")
    ascii_only = " ".join(ascii_only.split())
    return ascii_only.lower()


def canonicalise_class_name(name: str) -> Optional[str]:
    """Map a directory name to the canonical class name, if possible."""

    key = _normalise_string(name)
    return CLASS_ALIASES.get(key)


def iter_class_directories(root: Path) -> Iterable[Tuple[str, Path]]:
    """Yield ``(canonical_name, path)`` pairs for recognised class folders."""

    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        class_name = canonicalise_class_name(candidate.name)
        if class_name:
            yield class_name, candidate


def iter_image_files(directory: Path) -> Iterable[Path]:
    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        yield file_path


def build_image_index(roots: Sequence[str | Path], split: str) -> pd.DataFrame:
    """Scan each root directory and return a dataframe of image records.

    Parameters
    ----------
    roots:
        Sequence of dataset directories that contain class sub-folders.
    split:
        Name of the logical split represented by the directories (e.g. ``train``
        or ``val``). Stored in the dataframe for downstream filtering.
    """

    records: List[ImageRecord] = []

    for root in roots:
        base = Path(root).resolve()
        if not base.exists():
            raise FileNotFoundError(f"Dataset directory not found: {base}")

        for canonical_name, class_dir in iter_class_directories(base):
            label_id = CANONICAL_TO_ID[canonical_name]
            for image_path in iter_image_files(class_dir):
                records.append(
                    ImageRecord(
                        path=image_path.resolve(),
                        label_id=label_id,
                        label_name=canonical_name,
                        class_dir=class_dir.name,
                        split=split,
                    )
                )

    if not records:
        raise ValueError(
            "No images found. Ensure the dataset has recognised class folders."
        )

    data = {
        "path": [str(record.path) for record in records],
        "label_id": [record.label_id for record in records],
        "label_name": [record.label_name for record in records],
        "class_dir": [record.class_dir for record in records],
        "split": [record.split for record in records],
    }
    return pd.DataFrame(data)


def stratified_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int = 17,
    minimum_per_class: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/validation partitions with stratification."""

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    if val_fraction == 0.0:
        return df.copy(), df.iloc[0:0].copy()

    rng = random.Random(seed)
    val_indices: List[int] = []

    for label_id, group in df.groupby("label_id"):
        group_indices = list(group.index)
        if not group_indices:
            continue

        val_count = max(minimum_per_class, int(math.floor(len(group_indices) * val_fraction)))
        if val_count >= len(group_indices):
            val_count = max(1, len(group_indices) // 5)

        rng.shuffle(group_indices)
        val_indices.extend(group_indices[:val_count])

    val_df = df.loc[sorted(set(val_indices))].reset_index(drop=True)
    train_df = df.drop(val_indices).reset_index(drop=True)

    return train_df, val_df


def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor, image_size: Tuple[int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read an image from disk, resize, and scale pixels to [0, 1]."""

    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, image_size)
    return image, label


def make_tf_dataset(
    df: pd.DataFrame,
    batch_size: int,
    image_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    augment: Optional[tf.keras.Sequential] = None,
) -> tf.data.Dataset:
    """Build a ``tf.data.Dataset`` pipeline from a dataframe index."""

    paths = df["path"].values.astype("U")
    labels = df["label_id"].values.astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        buffer_size = max(len(df), batch_size * 4)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda p, y: _load_and_preprocess(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)

    if augment is not None:
        dataset = dataset.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def dataframe_from_dirs(train_dirs: Sequence[str | Path], val_dirs: Sequence[str | Path] | None = None) -> pd.DataFrame:
    """Convenience helper that concatenates indexes for train/val directories."""

    frames = [build_image_index(train_dirs, split="train")]

    if val_dirs:
        frames.append(build_image_index(val_dirs, split="val"))

    return pd.concat(frames, ignore_index=True)


__all__ = [
    "CANONICAL_LABELS",
    "CANONICAL_TO_ID",
    "ID_TO_CANONICAL",
    "build_image_index",
    "stratified_split",
    "make_tf_dataset",
    "dataframe_from_dirs",
]
