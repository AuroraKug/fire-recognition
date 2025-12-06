# Fire Recognition

## 项目简介
- 任务：火灾图像分类，支持 fire、no_fire、start_fire 三类。
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

## 训练入口（默认 FireCNN_v3）
- 脚本：`src/train.py`
- 功能：扫描 `dataset/FIRE_DATABASE_3`，划分训练/验证，训练 FireCNN_v3，记录日志并保存 `checkpoints/best.pth` 和 `checkpoints/last.pth`。
- 运行示例：
  ```bash
  python src/train.py
  ```
- 日志与可视化：
  - 文本日志：`src/logs/training_log_{timestamp}.txt`
  - TensorBoard：`src/logs/tensorboard/tb_{timestamp}/`
  - 曲线图：`python src/plot_logs.py` 自动生成 `src/figures/{timestamp}/loss_curve_{timestamp}.png` 与 `acc_curve_{timestamp}.png`

## 推理入口
- 脚本：`src/inference.py`
- 功能：加载指定 checkpoint（建议 `checkpoints/best.pth`），对 `dataset/TestData` 全部图像推理，生成 `src/outputs/result_{timestamp}.csv` 与 `src/outputs/result.csv`。
- 运行示例：
  ```bash
  python src/inference.py --checkpoint checkpoints/best.pth
  ```

## 模型与数据模块
- 数据集：`src/dataset.py` 定义 `FireDataset`，提供 `scan_dataset`、`scan_test`（不访问 TestData）。
- 模型：
  - `src/models/model_v1.py`：早期轻量 CNN
  - `src/models/model_v2.py`：SE 注意力版本
  - `src/models/model_v3.py`：强 stem + CBAM 注意力版本（训练与推理默认使用）
  - 接口一致：`build_model(num_classes, pretrained=False)`

## 重要说明
- 数据隔离：训练仅使用 `FIRE_DATABASE_3`，推理仅访问 `TestData`。
- 设备：自动选择 `cuda` 或 `cpu`，无需额外配置。
