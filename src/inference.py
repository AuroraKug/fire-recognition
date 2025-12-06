from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from torchvision import transforms

from dataset import LABEL_TO_INDEX
from models.model_v1 import build_model


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    '''
    Description
        构建推理阶段的transform 与验证阶段保持一致
    Args
        image_size (int): 输入图像统一尺寸
    Returns
        transform (transforms.Compose): 推理transform
    '''
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def load_images(root_dir: Path) -> List[Path]:
    '''
    Description
        扫描TestData目录下的全部图像路径
    Args
        root_dir (Path): 根目录应指向dataset/TestData
    Returns
        image_paths (List[Path]): 包含文件名后缀的完整图像路径列表
    '''
    subdirs = ["fire", "no_fire", "start_fire"]
    image_paths: List[Path] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for sub in subdirs:
        class_dir = root_dir / sub
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)
    if not image_paths:
        # 兼容无子目录时直接放置于TestData根目录的情况
        for path in sorted(root_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                image_paths.append(path)
    return image_paths


def predict(model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device) -> int:
    '''
    Description
        执行单张图像的前向推理
    Args
        model (torch.nn.Module): 已加载权重的模型
        img_tensor (torch.Tensor): 单张图像张量
        device (torch.device): 计算设备
    Returns
        pred_label (int): 预测标签编码
    '''
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor.to(device))
        pred = outputs.argmax(dim=1).item()
    return int(pred)


def write_csv(image_paths: Sequence[Path], predictions: Sequence[int], output_path: Path) -> None:
    '''
    Description
        将预测结果写入CSV文件
    Args
        image_paths (Sequence[Path]): 图像路径序列
        predictions (Sequence[int]): 对应的预测标签编码
        output_path (Path): CSV输出路径
    Returns
        None
    '''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ID,Label\n")
        for path, pred in zip(image_paths, predictions):
            f.write(f"{path.name},{pred}\n")


def main() -> None:
    '''
    Description
        加载模型与checkpoint 对TestData执行推理并输出CSV
    Args
        None
    Returns
        None
    '''
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--image_size", type=int, default=224, help="input image size")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent.parent / "dataset" / "TestData"), help="TestData root directory")
    parser.add_argument("--output", type=str, default=None, help="output csv path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_inference_transform(image_size=args.image_size)

    test_root = Path(args.data_root)
    image_paths = load_images(test_root)
    if len(image_paths) == 0:
        raise ValueError("No images found in TestData")

    num_classes = len(LABEL_TO_INDEX)
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    predictions: List[int] = []
    for path in image_paths:
        with path.open("rb") as f:
            img = Image.open(f).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        pred_label = predict(model, tensor, device)
        predictions.append(pred_label)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent / "outputs"
        output_path = output_dir / f"result_{timestamp}.csv"
    write_csv(image_paths, predictions, output_path)
    if not args.output:
        latest_path = output_path.parent / "result.csv"
        write_csv(image_paths, predictions, latest_path)


if __name__ == "__main__":
    main()
