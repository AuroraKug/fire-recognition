import math
from pathlib import Path
from typing import Tuple
from multiprocessing import cpu_count

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from dataset import FireDataset, LABEL_TO_INDEX, scan_dataset
from models.model_v4 import build_model
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


def build_transforms(image_size: int = 320) -> Tuple[transforms.Compose, transforms.Compose]:
    '''
    Description
        构建训练与验证阶段的transforms
    Args
        image_size (int): 输入图像的目标尺寸
    Returns
        transforms_pair (Tuple): 训练transform与验证transform
    '''
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transforms, val_transforms


def create_dataloaders(
    base_dir: Path,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    train_transforms: transforms.Compose,
    val_transforms: transforms.Compose,
) -> Tuple[DataLoader, DataLoader]:
    '''
    Description
        扫描FIRE_DATABASE_3并划分训练与验证集 构建对应的DataLoader
    Args
        base_dir (Path): 数据根目录包含FIRE_DATABASE_3
        batch_size (int): 每批次样本数量
        val_ratio (float): 验证集比例
        num_workers (int): DataLoader并行读取线程数
        train_transforms (transforms.Compose): 训练阶段预处理
        val_transforms (transforms.Compose): 验证阶段预处理
    Returns
        loaders (Tuple): 训练与验证DataLoader
    '''
    image_paths, labels = scan_dataset(base_dir)
    total = len(image_paths)
    if total == 0:
        raise ValueError("No training data found in FIRE_DATABASE_3")

    indices = torch.randperm(total).tolist()
    split_idx = int(total * (1 - val_ratio))
    split_idx = max(1, min(split_idx, total - 1))  # 确保训练与验证均非空
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = FireDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = FireDataset(val_paths, val_labels, transform=val_transforms)

    pin_device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device=pin_device,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        pin_memory_device=pin_device,
        persistent_workers=True,
        prefetch_factor=4,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    clip_norm: float = 5.0,
) -> Tuple[float, float]:
    '''
    Description
        执行单个epoch的训练计算loss与accuracy
    Args
        model (nn.Module): 训练模型
        dataloader (DataLoader): 训练数据迭代器
        criterion (nn.Module): 损失函数
        optimizer (optim.Optimizer): 优化器
        device (torch.device): 计算设备
        scaler (torch.amp.GradScaler): 混合精度缩放器
        clip_norm (float): 梯度裁剪阈值
    Returns
        metrics (Tuple): 平均loss与accuracy
    '''
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if clip_norm is not None and clip_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = running_correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    '''
    Description
        执行验证集评估计算loss与accuracy
    Args
        model (nn.Module): 待评估模型
        dataloader (DataLoader): 验证数据迭代器
        criterion (nn.Module): 损失函数
        device (torch.device): 计算设备
    Returns
        metrics (Tuple): 平均loss与accuracy
    '''
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validate", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            with torch.amp.autocast(device_type=device.type if device.type in {"cuda", "cpu"} else "cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = running_correct / total
    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, path: Path) -> None:
    '''
    Description
        保存模型权重到指定路径
    Args
        model (nn.Module): 需要保存的模型
        path (Path): 输出文件路径
    Returns
        None: 无返回值
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    '''
    Description
        组织训练全流程 初始化模型 transforms dataloader 并执行训练与验证
    Args
        None
    Returns
        None
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent.parent / "dataset"

    num_classes = len(LABEL_TO_INDEX)
    image_size = 320
    batch_size = 64
    num_epochs = 150
    val_ratio = 0.2
    try:
        num_workers = min(16, cpu_count())
    except Exception:
        num_workers = 8
    print(f"Using num_workers={num_workers}")
    learning_rate = 0.0025
    weight_decay = 3e-4
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    log_dir = Path(__file__).resolve().parent / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = log_dir / "tensorboard" / f"tb_{timestamp}"
    txt_log_path = log_dir / f"training_log_{timestamp}.txt"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    train_transforms, val_transforms = build_transforms(image_size=image_size)
    train_loader, val_loader = create_dataloaders(
        base_dir=base_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )

    try:
        model = build_model(num_classes=num_classes, pretrained=False)
    except TypeError:
        model = build_model(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0

    writer = SummaryWriter(log_dir=tb_dir)
    with txt_log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                f"LR: {current_lr:.6f}"
            )

            log_line = (
                f"Epoch {epoch + 1} "
                f"TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} "
                f"ValLoss {val_loss:.4f} ValAcc {val_acc:.4f} "
                f"LR {current_lr:.6f}"
            )
            log_f.write(log_line + "\n")
            writer.add_scalar("Loss/train", train_loss, epoch + 1)
            writer.add_scalar("Loss/val", val_loss, epoch + 1)
            writer.add_scalar("Acc/train", train_acc, epoch + 1)
            writer.add_scalar("Acc/val", val_acc, epoch + 1)
            writer.add_scalar("LR", current_lr, epoch + 1)

            save_checkpoint(model, checkpoint_dir / "last.pth")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, checkpoint_dir / "best.pth")
            print(f"Early-stopping monitor (log only): best_val_acc={best_val_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
