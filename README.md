# Fire Detection from Scratch

## 方法介绍
- **数据处理**：使用 `ImageDataGenerator` 做标准化和增强（旋转、平移、缩放、剪切、亮度、通道偏移、水平翻转）。输入分辨率 224×224，未使用 MixUp/CutMix（最佳实验关闭）。
- **模型架构**：自定义轻量级 Inception-like CNN（自建 stem + Inception-A/B 与 reduction 模块），无预训练；头部为 GAP + Dropout + 全连接 softmax。
- **损失与正则**：交叉熵 + label smoothing 0.05，卷积与头部均匀 L2 正则（1e-4），Dropout(0.3 头部)。

## 实验设置
- **环境**：Python 3.10；GPU: RTX 3090；TensorFlow 2.x（oneDNN/CUDA 启用）；主要依赖：`tensorflow`、`pandas`、`numpy`、`Pillow`、`scikit-learn`。
- **超参数（最佳 0.878 提交）**：
  - 模型：`--model_type inception`
  - 输入：`--img_size 224`
  - 批大小：`--batch_size 16`
  - 优化器：Adam，初始 LR `2e-4`
  - 学习率策略：`--lr_schedule plateau`（ReduceLROnPlateau 监控 val_loss，factor 0.5，patience 6，min_lr 1e-7）
  - 早停：`--patience 18`（monitor val_loss，restore best weights）
  - 正则：`--l2_weight 1e-4`
  - 数据：独立 train/val 目录；未使用 MixUp/CutMix（`--no_mixup`），未使用类权重（因未开启 MixUp/CutMix 时 class_weight 可用，但本次关闭）。
  - 轮次：`--epochs 90`（早停于最佳轮次）

## 结果分析
- 训练日志显示初期收敛后，val_accuracy 在 0.78 附近震荡，经 LR 降阶与长耐心稳定提升，最佳 val_accuracy ≈ 0.853，线上提交得分 0.878。
- Val loss 在 0.8–1.3 间波动，L2+label smoothing 帮助抑制过拟合；Plateau 调度在中后期提供小幅提升。

## 数据准备
- 目录结构示例：
  ```
  datasets/
    train/
      fire/
      no_fire/
      start_fire/
    val/
      fire/
      no_fire/
      start_fire/
    test/
      *.jpg|png
  ```
- 如路径不同，可通过 `--train_root`、`--val_root`、`--test_root` 传入自定义位置。

## 训练步骤
- 使用最佳设置训练：
  ```bash
  python train.py \
    --train_root datasets/train --val_root datasets/val \
    --output_dir models_v5_incept_ce_plateau224 \
    --epochs 90 --batch_size 16 \
    --learning_rate 2e-4 \
    --model_type inception \
    --lr_schedule plateau \
    --no_mixup \
    --img_size 224 \
    --l2_weight 1e-4 \
    --patience 18
  ```
- 训练产物：`best_model.keras`、`final_model.keras`、`classes.txt`、`training_history.json` 保存在 `--output_dir`。

## 推理与生成提交
- 使用最佳模型推理并生成提交 CSV：
  ```bash
  python inference.py \
    --model_path models_v5_incept_ce_plateau224/best_model.keras \
    --test_root datasets/test \
    --output_csv submission.csv \
    --classes_file models_v5_incept_ce_plateau224/classes.txt
  ```
- 输出仅含 `ID,Label` 列，按文件名排序，适配比赛提交格式。

## 备注与改进方向
- 若显存允许，可尝试更大批次并线性放大学习率；或轻度 MixUp/CutMix 做额外模型并行系数投票融合。
- 进一步可调：提升/降低 L2、尝试 cosine+warmup、或对 “start_fire” 类做轻度过采样以平衡召回。
