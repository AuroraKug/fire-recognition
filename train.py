import argparse
import os
import sys
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    Activation,
    Add,
    SeparableConv2D,
    AveragePooling2D,
    MaxPooling2D,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, losses

# 将项目根目录添加到路径中，以便导入 dataset 的 utilities
sys.path.append(str(Path(__file__).parent))
from dataset import build_image_index, stratified_split

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 90
LEARNING_RATE = 3e-4

# 深度可分离残差块，可在不增加大量参数的情况下提高模型容量
def residual_block(x, filters, stride=1, dropout_rate=0.0, l2_weight=1e-4):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_weight),
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = SeparableConv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(l2_weight),
        pointwise_regularizer=regularizers.l2(l2_weight),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(l2_weight),
        pointwise_regularizer=regularizers.l2(l2_weight),
    )(x)
    x = BatchNormalization()(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)
    return x


def conv_bn_relu(x, filters, kernel_size, strides=1, padding="same", l2_weight=5e-5):
    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_weight),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def inception_a(x, filters, l2_weight=5e-5):
    b1 = conv_bn_relu(x, filters, 1, l2_weight=l2_weight)

    b2 = conv_bn_relu(x, filters, 1, l2_weight=l2_weight)
    b2 = conv_bn_relu(b2, filters, 3, l2_weight=l2_weight)

    b3 = conv_bn_relu(x, filters, 1, l2_weight=l2_weight)
    b3 = conv_bn_relu(b3, filters, 5, l2_weight=l2_weight)

    b4 = AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters, 1, l2_weight=l2_weight)

    return Concatenate()([b1, b2, b3, b4])


def reduction_a(x, k=128, l=128, m=192, n=224, l2_weight=5e-5):
    b1 = conv_bn_relu(x, n, 3, strides=2, padding="valid", l2_weight=l2_weight)

    b2 = conv_bn_relu(x, k, 1, l2_weight=l2_weight)
    b2 = conv_bn_relu(b2, l, 3, l2_weight=l2_weight)
    b2 = conv_bn_relu(b2, m, 3, strides=2, padding="valid", l2_weight=l2_weight)

    b3 = MaxPooling2D(3, strides=2, padding="valid")(x)

    return Concatenate()([b1, b2, b3])


def inception_b(x, filters, l2_weight=5e-5):
    b1 = conv_bn_relu(x, filters, 1, l2_weight=l2_weight)

    b2 = conv_bn_relu(x, filters, 1, l2_weight=l2_weight)
    b2 = conv_bn_relu(b2, filters, (1, 3), l2_weight=l2_weight)
    b2 = conv_bn_relu(b2, filters, (3, 1), l2_weight=l2_weight)

    b3 = conv_bn_relu(x, filters, 1, l2_weight=l2_weight)
    b3 = conv_bn_relu(b3, filters, 3, l2_weight=l2_weight)
    b3 = conv_bn_relu(b3, filters, (1, 3), l2_weight=l2_weight)
    b3 = conv_bn_relu(b3, filters, (3, 1), l2_weight=l2_weight)

    b4 = AveragePooling2D(3, strides=1, padding="same")(x)
    b4 = conv_bn_relu(b4, filters, 1, l2_weight=l2_weight)

    return Concatenate()([b1, b2, b3, b4])


def reduction_b(x, l2_weight=5e-5):
    b1 = conv_bn_relu(x, 192, 1, l2_weight=l2_weight)
    b1 = conv_bn_relu(b1, 224, (1, 3), l2_weight=l2_weight)
    b1 = conv_bn_relu(b1, 256, (3, 1), l2_weight=l2_weight)
    b1 = conv_bn_relu(b1, 256, 3, strides=2, padding="valid", l2_weight=l2_weight)

    b2 = conv_bn_relu(x, 192, 1, l2_weight=l2_weight)
    b2 = conv_bn_relu(b2, 192, 3, strides=2, padding="valid", l2_weight=l2_weight)

    b3 = MaxPooling2D(3, strides=2, padding="valid")(x)

    return Concatenate()([b1, b2, b3])


def create_inception_like(num_classes, l2_weight=5e-5, dropout_head=0.3):
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = conv_bn_relu(inputs, 32, 3, strides=2, padding="valid", l2_weight=l2_weight)
    x = conv_bn_relu(x, 32, 3, padding="valid", l2_weight=l2_weight)
    x = conv_bn_relu(x, 64, 3, l2_weight=l2_weight)
    x = MaxPooling2D(3, strides=2, padding="valid")(x)

    x = conv_bn_relu(x, 80, 1, padding="valid", l2_weight=l2_weight)
    x = conv_bn_relu(x, 96, 3, padding="valid", l2_weight=l2_weight)
    x = MaxPooling2D(3, strides=2, padding="valid")(x)

    x = inception_a(x, 48, l2_weight)
    x = inception_a(x, 56, l2_weight)
    x = reduction_a(x, l2_weight=l2_weight)

    x = inception_b(x, 64, l2_weight)
    x = inception_b(x, 72, l2_weight)
    x = reduction_b(x, l2_weight)

    x = inception_b(x, 80, l2_weight)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_head)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, model


def create_residual_model(num_classes, l2_weight=5e-5):
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        strides=1,
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_weight),
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        strides=2,
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_weight),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = residual_block(x, 64, stride=2, dropout_rate=0.05, l2_weight=l2_weight)
    x = residual_block(x, 64, l2_weight=l2_weight)
    x = residual_block(x, 96, stride=2, dropout_rate=0.1, l2_weight=l2_weight)
    x = residual_block(x, 96, l2_weight=l2_weight)
    x = residual_block(x, 128, stride=2, dropout_rate=0.15, l2_weight=l2_weight)

    x = SeparableConv2D(
        192,
        (3, 3),
        padding="same",
        use_bias=False,
        depthwise_regularizer=regularizers.l2(l2_weight),
        pointwise_regularizer=regularizers.l2(l2_weight),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(192, activation="relu", kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = Dropout(0.25)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model, model


@tf.keras.utils.register_keras_serializable(package="Custom")
class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay + linear warmup."""

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

def apply_mixup(x_batch, y_batch, alpha=0.2, prob=0.5):
    if np.random.rand() > prob:
        return x_batch, y_batch

    lam = np.random.beta(alpha, alpha)
    batch_size = x_batch.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * x_batch + (1 - lam) * x_batch[index]
    mixed_y = lam * y_batch + (1 - lam) * y_batch[index]
    return mixed_x, mixed_y


def apply_cutmix(x_batch, y_batch, alpha=0.6, prob=0.3):
    if np.random.rand() > prob:
        return x_batch, y_batch

    lam = np.random.beta(alpha, alpha)
    batch_size, h, w, _ = x_batch.shape
    index = np.random.permutation(batch_size)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    for i in range(batch_size):
        x_batch[i, y1:y2, x1:x2, :] = x_batch[index[i], y1:y2, x1:x2, :]

    box_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1.0 - box_area / float(h * w)
    mixed_y = lam_adjusted * y_batch + (1.0 - lam_adjusted) * y_batch[index]
    return x_batch, mixed_y


def create_data_generators(train_df, val_df, class_names, batch_size, use_mixup=True, use_cutmix=False):
    train_datagen = ImageDataGenerator(
        rescale=1./255, # 将像素值缩放到 [0, 1]
        rotation_range=25,  # 随机旋转图像
        width_shift_range=0.2,  # 随机水平平移
        height_shift_range=0.2, # 随机垂直平移
        shear_range=0.2,    # 随机剪切
        zoom_range=0.25,    # 随机缩放
        brightness_range=(0.75, 1.25),  # 随机调整亮度
        channel_shift_range=25.0,  # 随机通道偏移
        horizontal_flip=True,  # 随机水平翻转
        fill_mode='nearest'  # 填充空白区域
    )

    val_datagen = ImageDataGenerator(rescale=1./255)  # 归一化

    base_train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=True
    )

    def train_generator():
        for x_batch, y_batch in base_train_gen:
            if use_cutmix:
                x_batch, y_batch = apply_cutmix(x_batch, y_batch, alpha=0.6, prob=0.3)
            if use_mixup:
                x_batch, y_batch = apply_mixup(x_batch, y_batch, alpha=0.2, prob=0.6)
            yield x_batch, y_batch

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )

    if use_mixup:
        train_steps = len(base_train_gen)
        val_steps = len(val_generator)
        return train_generator(), val_generator, train_steps, val_steps
    else:
        # return the Sequence iterator directly to allow class_weight usage
        train_steps = len(base_train_gen)
        val_steps = len(val_generator)
        return base_train_gen, val_generator, train_steps, val_steps


def compute_class_weights(train_df, class_names):
    """计算类别权重。使用平方根减缓权重的极端变化"""

    counts = train_df['class'].value_counts()
    max_count = counts.max()
    return {class_names.index(cls): float((max_count / counts[cls]) ** 0.5) for cls in class_names}

def main():
    parser = argparse.ArgumentParser(description='Train fire detection model')
    parser.add_argument('--train_root', required=True, help='Root directory of training data')
    parser.add_argument('--val_root', help='Root directory of validation data (optional)')
    parser.add_argument('--output_dir', default='models', help='Output directory for saved models')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Fraction of training data to use for validation if val_root not provided')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss instead of cross-entropy')
    parser.add_argument('--model_type', choices=['residual', 'inception'], default='residual', help='Backbone type')
    parser.add_argument('--no_mixup', action='store_true', help='Disable mixup augmentation')
    parser.add_argument('--use_cutmix', action='store_true', help='Enable CutMix augmentation (disables class_weight)')
    parser.add_argument('--img_size', type=int, default=224, help='Square input resolution (e.g., 224, 256, 288)')
    parser.add_argument('--lr_schedule', choices=['plateau', 'cosine'], default='plateau', help='Learning rate schedule type')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs for cosine schedule')
    parser.add_argument('--l2_weight', type=float, default=1e-4, help='L2 weight decay for conv layers and head')
    parser.add_argument('--patience', type=int, default=18, help='Early stopping patience')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global IMG_HEIGHT, IMG_WIDTH
    IMG_HEIGHT = args.img_size
    IMG_WIDTH = args.img_size

    print("Building image index...")
    # Build image index from training data
    train_df = build_image_index([args.train_root], split="train")

    if train_df.empty:
        print(f"No images found in {args.train_root}")
        return

    # Get class names
    class_names = sorted(train_df['label_name'].unique())
    num_classes = len(class_names)
    print(f"Found classes: {class_names}")

    # Split data
    if args.val_root:
        print("Using separate validation set...")
        val_files_df = build_image_index([args.val_root], split="val")
        train_files_df = train_df
    else:
        print(f"Splitting training data with test_split={args.test_split}...")
        train_files_df, val_files_df = stratified_split(train_df, val_fraction=args.test_split)

    print(f"Training samples: {len(train_files_df)}")
    print(f"Validation samples: {len(val_files_df)}")

    # 重命名列名以兼容 ImageDataGenerator
    train_files_df = train_files_df.rename(columns={'path': 'filepath', 'label_name': 'class'})
    val_files_df = val_files_df.rename(columns={'path': 'filepath', 'label_name': 'class'})

    class_weights = compute_class_weights(train_files_df, class_names)
    print(f"Class weights: {class_weights}")

    # create data generators
    use_mixup = not args.no_mixup
    use_cutmix = args.use_cutmix
    train_generator, val_generator, train_steps, val_steps = create_data_generators(
        train_files_df,
        val_files_df,
        class_names,
        batch_size=args.batch_size,
        use_mixup=use_mixup,
        use_cutmix=use_cutmix,
    )

    if use_mixup or use_cutmix:
        class_weight_arg = None
        print("MixUp/CutMix enabled: disabling class_weight (not supported with Python generator).")
    else:
        class_weight_arg = class_weights

    print("Creating model...")
    if args.model_type == 'inception':
        model, base_model = create_inception_like(num_classes, l2_weight=args.l2_weight)
    else:
        model, base_model = create_residual_model(num_classes, l2_weight=args.l2_weight)

    # Loss: label smoothing or focal loss based on flag
    if args.use_focal_loss:
        def focal_loss(y_true, y_pred, gamma=1.5, alpha=0.75):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weights = alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_mean(tf.reduce_sum(weights * cross_entropy, axis=1))

        loss_fn = focal_loss
    else:
        loss_fn = losses.CategoricalCrossentropy(label_smoothing=0.05)

    if args.lr_schedule == 'cosine':
        total_steps = train_steps * args.epochs
        warmup_steps = args.warmup_epochs * train_steps
        lr_schedule = WarmupCosineSchedule(args.learning_rate, total_steps, warmup_steps)
        optimizer = Adam(learning_rate=lr_schedule)
        print(f"Using cosine LR with warmup: base_lr={args.learning_rate}, warmup_steps={warmup_steps}, total_steps={total_steps}")
    else:
        optimizer = Adam(learning_rate=args.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint_path = output_dir / 'best_model.keras'
    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

    if args.lr_schedule == 'plateau':
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                verbose=1
            )
        )

    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        class_weight=class_weight_arg,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = output_dir / 'final_model.keras'
    model.save(str(final_model_path))
    print(f"Final model saved to {final_model_path}")

    # Save training history
    import json
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Save class names
    classes_path = output_dir / 'classes.txt'
    with open(classes_path, 'w') as f:
        f.write('\n'.join(class_names))
    print(f"Class names saved to {classes_path}")

    print("Training completed!")

if __name__ == '__main__':
    main()