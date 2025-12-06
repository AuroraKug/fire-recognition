from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import FireDataset, LABEL_TO_INDEX, scan_dataset
from models.model_v1 import build_model


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
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
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
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
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

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
    image_size = 224
    batch_size = 32
    num_epochs = 20
    val_ratio = 0.2
    num_workers = 4
    learning_rate = 1e-3
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        save_checkpoint(model, checkpoint_dir / "last.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_dir / "best.pth")


if __name__ == "__main__":
    main()
