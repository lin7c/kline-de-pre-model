import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import idct

def draw_candlesticks(ax, ohlc_data, x_offset=0, width=0.6, color_up='red', color_down='green'):
    for i in range(len(ohlc_data)):
        open_p, high_p, low_p, close_p = ohlc_data[i]
        x = i + x_offset
        color = color_up if close_p >= open_p else color_down
        ax.vlines(x, low_p, high_p, color=color, linewidth=1)
        rect_height = abs(close_p - open_p)
        if rect_height == 0:
            rect_height = 0.0001
        rect_bottom = min(open_p, close_p)
        ax.add_patch(plt.Rectangle((x - width / 2, rect_bottom), width, rect_height,
                                   facecolor=color, edgecolor=color))


def plot_kline_comparison(idx, delta_path, dct_path, x_path):
    # 1. 加载数据
    Y_delta = np.load(delta_path)   # (N, 60, 4)   ← 已经标准化过的 Delta
    Y_dct = np.load(dct_path)       # (N, 9)
    X_raw = np.load(x_path)         # (N, 60, 12)

    # ====================== 必须修改的部分 ======================
    # 计算 local_std（与 make_delta_data 中完全一致）
    local_std = np.std(X_raw[idx, :, 3]) + 1e-9

    # 重建 DCT 趋势线（必须乘回 local_std）
    coeffs_1m = Y_dct[idx, :3]
    full_coeffs = np.zeros(60)
    full_coeffs[:3] = coeffs_1m
    trend_line = idct(full_coeffs, type=2, norm='ortho').reshape(60, 1) * local_std

    base_price = X_raw[idx, -1, 3]

    # 重建 OHLC（对 Delta 进行逆标准化）
    reconstructed_ohlc = (Y_delta[idx] * local_std) + trend_line + base_price

    # 真实未来 OHLC
    true_ohlc = X_raw[idx + 1: idx + 61, -1, :4]

    # 计算重建误差（强烈建议保留，用于验证）
    mae = np.mean(np.abs(reconstructed_ohlc - true_ohlc))
    print(f"Sample {idx} | local_std = {local_std:.4f} | 重建 MAE = {mae:.6f}")

    # ====================== 绘图 ======================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)

    draw_candlesticks(ax1, true_ohlc)
    ax1.plot(range(60), trend_line + base_price, color='blue', linestyle='--', alpha=0.8, label='DCT Trend Line')
    ax1.set_title(f"Sample {idx}: True Future Candlesticks", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    draw_candlesticks(ax2, reconstructed_ohlc)
    ax2.plot(range(60), trend_line + base_price, color='blue', linestyle='--', alpha=0.8, label='DCT Trend Line')
    ax2.set_title(f"Sample {idx}: Reconstructed (Trend + Delta) | local_std={local_std:.4f}", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.xlabel("Future Time Steps (1-60)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_kline_comparison(
        idx=200,
        delta_path="y_delta_ohlc.npy",
        dct_path="../CNN_Transformer/y_transformer_v1.npy",
        x_path="../CNN/input_x_v1.npy"
    )