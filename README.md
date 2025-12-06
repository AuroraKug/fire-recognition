# Fire Recognition

English version follows Chinese instructions.

## 简要说明
- 项目目标：火灾图像分类，训练自定义轻量CNN。
- 数据目录：`dataset/`
  - 训练集：`dataset/FIRE_DATABASE_3/{fire,no_fire,start_fire}`
  - 测试集：`dataset/test`（仅评估，不参与训练）
  - 最终评估：`dataset/TestData/{fire,no_fire,start_fire}`（仅推理）

## 环境准备
1. 创建并激活虚拟环境（可选）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   ```
2. 安装依赖：
   ```bash
   pip install -r src/requirements.txt
   ```

## 代码入口与用途
- 训练：`src/train.py`
  - 功能：扫描 `dataset/FIRE_DATABASE_3`，划分训练/验证，训练模型，保存 `checkpoints/best.pth` 与 `checkpoints/last.pth`。
  - 运行示例：
    ```bash
    python src/train.py
    ```
- 推理：`src/inference.py`
  - 功能：加载指定 checkpoint，对 `dataset/TestData` 全部图像推理，生成 `src/outputs/result_{timestamp}.csv` 和 `src/outputs/result.csv`。
  - 运行示例：
    ```bash
    python src/inference.py --checkpoint checkpoints/best.pth
    ```

## 模型与数据模块
- 数据集定义：`src/dataset.py` 提供 `FireDataset` 及扫描训练/测试集的工具函数（不触碰 TestData）。
- 模型定义：`src/models/model_v1.py` 提供轻量 CNN，`build_model(num_classes, pretrained=False)`。

## 其他
- 保持数据隔离：训练仅使用 `FIRE_DATABASE_3`；推理仅访问 `TestData`。
- 需要 GPU 时自动选择 `cuda`，否则使用 `cpu`。*** End Patch***" json-pointer="/chat.completion/assistant/18/content" zod="apply_patch_schema/2.0"/> 🍃 Object schema validation 🛡️ ***!
