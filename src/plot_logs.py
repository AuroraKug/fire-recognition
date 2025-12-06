from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # 关键导入 用于强制整数刻度

def _find_latest_log(log_dir: Path) -> Path:
    '''
    Description
        查找日志目录下最新的训练日志文件
    Args
        log_dir (Path): 日志目录
    Returns
        log_path (Path): 最新日志文件路径
    '''
    # 确保目录存在
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
        
    log_files = sorted(log_dir.glob("training_log_*.txt"))
    if not log_files:
        raise FileNotFoundError("No training log found in logs directory")
    return log_files[-1]


def _parse_log(log_path: Path) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    '''
    Description
        解析训练日志提取指标序列
    Args
        log_path (Path): 日志文件路径
    Returns
        metrics (Tuple): 包含epoch train_loss train_acc val_loss val_acc
    '''
    epochs: List[int] = []
    train_loss: List[float] = []
    train_acc: List[float] = []
    val_loss: List[float] = []
    val_acc: List[float] = []
    
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # 简单的健壮性检查 防止空行报错
            if len(parts) < 10: 
                continue
            try:
                # 假设格式固定 根据原代码逻辑提取
                epoch_idx = int(parts[1])
                t_loss = float(parts[3])
                t_acc = float(parts[5])
                v_loss = float(parts[7])
                v_acc = float(parts[9])
            except (ValueError, IndexError):
                continue
            
            epochs.append(epoch_idx)
            train_loss.append(t_loss)
            train_acc.append(t_acc)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            
    if not epochs:
        raise ValueError("Log file is empty or format mismatch")
    return epochs, train_loss, train_acc, val_loss, val_acc


def _plot_curve(
    epochs: List[int],
    train_vals: List[float],
    val_vals: List[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    '''
    Description
        绘制训练与验证曲线并保存
    Args
        epochs (List[int]): 训练轮次
        train_vals (List[float]): 训练指标
        val_vals (List[float]): 验证指标
        ylabel (str): y轴名称
        title (str): 图标题
        output_path (Path): 输出文件路径
    Returns
        None
    '''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 开始绘图设置
    # 设置白色背景和网格风格
    plt.style.use('default')  # 重置为默认 避免之前的干扰
    
    # 创建画布 设置大小和分辨率
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # 绘制曲线
    # Train 使用深蓝色 带圆点标记
    ax.plot(epochs, train_vals, label="Train", color='#2878B5', 
            linewidth=2.5, marker='o', markersize=6, linestyle='-')
    
    # Val 使用橙红色 带方块标记
    ax.plot(epochs, val_vals, label="Validation", color='#C82423', 
            linewidth=2.5, marker='s', markersize=6, linestyle='--')

    # 关键修复 强制X轴为整数
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置标题和标签 字体加大
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Epoch", fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)

    # 刻度字体调整
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 美化图例
    ax.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9, loc='best')

    # 美化网格 灰色虚线 置于底层
    ax.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=0)

    # 移除顶部和右侧的边框 让图表更清爽
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() # 关闭图形释放内存


def main() -> None:
    '''
    Description
        读取最新训练日志并生成loss与accuracy曲线
    Args
        None
    Returns
        None
    '''
    # 获取当前脚本所在目录
    src_dir = Path(__file__).resolve().parent
    log_dir = src_dir / "logs"
    figures_dir = src_dir / "figures"

    try:
        log_path = _find_latest_log(log_dir)
        print(f"Processing log: {log_path.name}")
        
        epochs, train_loss, train_acc, val_loss, val_acc = _parse_log(log_path)
        
        # 提取时间戳用于命名
        timestamp = log_path.stem.replace("training_log_", "", 1)
        out_dir = figures_dir / timestamp
        
        loss_path = out_dir / f"loss_curve_{timestamp}.png"
        acc_path = out_dir / f"acc_curve_{timestamp}.png"

        _plot_curve(epochs, train_loss, val_loss, ylabel="Loss", title="Training & Validation Loss", output_path=loss_path)
        _plot_curve(epochs, train_acc, val_acc, ylabel="Accuracy", title="Training & Validation Accuracy", output_path=acc_path)
        
        print(f"Figures saved to: {out_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
