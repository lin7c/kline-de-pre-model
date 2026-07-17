import numpy as np
from scipy.fftpack import dct
import os


def generate_dct_trend_labels(file_path, n_components=3):
    """v3: 只回归 1m 未来 60 分钟趋势的前 n_components 个 DCT 系数(局部标准化空间)。

    v2 的 9 维(3 周期×3 系数)已改为 3 维: 5m/15m 的趋势输出被砍掉——
    walk-forward 探针显示 300/900 分钟前向信息更弱, 多任务头徒增拟合目标。
    保留 total-900 的裁剪跨度, 与 CNN/makedata 的 input_x 对齐。
    """
    try:
        X_raw = np.load(file_path)
    except FileNotFoundError:
        print(f"未找到数据文件: {file_path}")
        return None

    total_samples = X_raw.shape[0]
    gap_align = 900          # 与 input_x 裁剪对齐(管线最大前视跨度)
    look_ahead = 60          # 1m 趋势预测跨度
    col_idx = 3              # 1m close

    valid_samples = total_samples - gap_align
    if valid_samples <= 0:
        print("数据量太小，无法生成标签")
        return None

    closes_end = X_raw[:, -1, col_idx].astype(np.float64)             # 每样本右端收盘 (T,)
    local_std = X_raw[:, :, col_idx].astype(np.float64).std(axis=1) + 1e-9  # 每窗口 σ (T,)

    print("开始生成带局部标准化的 DCT 趋势标签(v3: 1m×3 系数)...")
    idx = np.arange(valid_samples)
    # 未来 60 根收盘: fut[i, k] = closes_end[i+1+k]
    fut = closes_end[idx[:, None] + 1 + np.arange(look_ahead)[None, :]]
    norm = (fut - closes_end[idx, None]) / local_std[idx, None]
    coeffs = dct(norm, type=2, norm="ortho", axis=1)[:, :n_components]
    return coeffs.astype(np.float32)


def run(X_FILE="../org_v1.npy", Y_FILE="y_transformer_v1.npy"):
    y = generate_dct_trend_labels(X_FILE, n_components=3)
    if y is not None:
        np.save(Y_FILE, y)
        print(f"处理完成！最终 Y 形状: {y.shape}")
        print(f"Y 统计: mean={y.mean():.4f}, std={y.std():.4f}")


if __name__ == "__main__":
    run()
