import numpy as np
from scipy.fftpack import dct
import os


def generate_dct_trend_labels(file_path, n_components=3):
    try:
        X_raw = np.load(file_path)
    except FileNotFoundError:
        print(f"未找到数据文件: {file_path}")
        return None

    total_samples = X_raw.shape[0]

    gap_1m = 60
    gap_5m = 300
    gap_15m = 900

    valid_samples = total_samples - gap_15m

    if valid_samples <= 0:
        print(f"数据量太小，无法生成标签")
        return None

    config = [
        {"col": 3, "gap": gap_1m, "step": 1},
        {"col": 7, "gap": gap_5m, "step": 5},
        {"col": 11, "gap": gap_15m, "step": 15}
    ]

    y_list = []
    print(f"开始生成带局部标准化的 DCT 趋势标签...")

    for i in range(valid_samples):
        sample_all_tf_coeffs = []

        for item in config:
            col_idx = item["col"]
            look_ahead = item["gap"]
            step = item["step"]

            # 1. 提取基准价格
            base_price = X_raw[i, -1, col_idx]

            # --- 新增：计算局部标准化因子 ---
            # 使用当前观察窗口 X[i] 的标准差作为缩放基准
            # 加上 1e-9 防止除零
            local_std = np.std(X_raw[i, :, col_idx]) + 1e-9

            # 2. 提取未来原始序列并压缩
            future_series_raw = X_raw[i + 1: i + look_ahead + 1, -1, col_idx]
            compressed_series = future_series_raw[step - 1::step]

            # 3. --- 修改：执行局部标准化 ---
            # 原来的：norm_series = compressed_series - base_price
            # 现在的：(未来价格 - 当前价格) / 历史波动波动
            norm_series = (compressed_series - base_price) / local_std

            # 4. 执行 DCT 变换
            coeffs = dct(norm_series, type=2, norm='ortho')

            # 5. 截取前 n_components 个低频系数
            sample_all_tf_coeffs.append(coeffs[:n_components])

        y_list.append(np.concatenate(sample_all_tf_coeffs))

    y_final = np.array(y_list, dtype=np.float32)
    return y_final


def run(X_FILE="../org_v1.npy", Y_FILE="y_transformer_v1.npy"):
    y = generate_dct_trend_labels(X_FILE, n_components=3)
    if y is not None:
        np.save(Y_FILE, y)
        print(f"处理完成！最终 Y 形状: {y.shape}")


if __name__ == "__main__":
    run()