from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset

# 固定的标签编码映射 使用英文字段避免训练测试不一致
LABEL_TO_INDEX = {"fire": 0, "no_fire": 1, "start_fire": 2}
# 支持的图像扩展名 仅筛选可读取的图像文件
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class FireDataset(Dataset):
    '''
    类描述
        用于火灾图像分类数据加载 接收路径和标签并应用外部transform
    Args
        image_paths (Sequence[Path]): 图像路径序列
        labels (Sequence[int]): 标签编码序列
        transform (Callable): 预处理函数由训练脚本提供
    '''
    def __init__(self, image_paths: Sequence[Path], labels: Sequence[int], transform: Callable):
        if transform is None:
            raise ValueError("transform is required to enforce resize and normalize outside dataset.py")
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = [int(l) for l in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    '''
    Description
        按索引读取图像并返回标签编码
    Args
        idx (int): 样本索引
    Returns
        item (Tuple): 图像张量与标签编码
    '''
    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        with image_path.open("rb") as f:
            image = Image.open(f).convert("RGB")
        image = self.transform(image) if self.transform else image
        return image, label


def _gather_class_images(root: Path, class_name: str) -> List[Path]:
    # 收集指定类别目录下的全部图像路径
    class_dir = root / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Expected class directory missing: {class_dir}")
    images: List[Path] = []
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def scan_dataset(base_dir: Path) -> Tuple[List[Path], List[int]]:
    '''
    Description
        扫描FIRE_DATABASE_3目录获取训练集路径与标签 禁止包含TestData目录
    Args
        base_dir (Path): 根目录包含FIRE_DATABASE_3
    Returns
        dataset_info (Tuple): 训练集图像路径列表与标签编码列表
    '''
    root = Path(base_dir) / "FIRE_DATABASE_3"
    if not root.exists():
        raise FileNotFoundError(f"Training directory not found: {root}")

    image_paths: List[Path] = []
    labels: List[int] = []
    for class_name, encoded_label in LABEL_TO_INDEX.items():
        class_images = _gather_class_images(root, class_name)
        image_paths.extend(class_images)
        labels.extend([encoded_label] * len(class_images))
    return image_paths, labels


def scan_test(base_dir: Path) -> Tuple[List[Path], List[int]]:
    '''
    Description
        扫描test目录获取测试集路径与标签 禁止扫描TestData以防数据泄漏
    Args
        base_dir (Path): 根目录包含test
    Returns
        test_info (Tuple): 测试集图像路径列表与标签编码列表
    '''
    root = Path(base_dir) / "test"
    if not root.exists():
        raise FileNotFoundError(f"Test directory not found: {root}")

    image_paths: List[Path] = []
    labels: List[int] = []
    for class_name, encoded_label in LABEL_TO_INDEX.items():
        class_images = _gather_class_images(root, class_name)
        image_paths.extend(class_images)
        labels.extend([encoded_label] * len(class_images))
    return image_paths, labels
