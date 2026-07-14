import numpy as np
from scipy.fftpack import idct


def make_delta_data(input_x_file="../CNN/input_x_v1.npy",
                    dct_coeff_file="../CNN_Transformer/y_transformer_v1.npy",
                    output_delta_file="y_delta_ohlc.npy"):
    X_raw = np.load(input_x_file)
    dct_coeffs = np.load(dct_coeff_file)

    look_ahead = 60
    final_valid_len = min(X_raw.shape[0], len(dct_coeffs)) - look_ahead

    delta_list = []
    print(f"正在剥离趋势，提取残差细节（已加入局部标准化）...")

    for i in range(final_valid_len):
        # === 1. 计算 local_std（与 DCT 生成时完全一致）===
        local_std = np.std(X_raw[i, :, 3]) + 1e-9

        # === 2. 重建 DCT 趋势线 ===
        coeffs_1m = dct_coeffs[i, :3]
        full_coeffs = np.zeros(look_ahead)
        full_coeffs[:3] = coeffs_1m
        trend_line = idct(full_coeffs, type=2, norm='ortho').reshape(60, 1)

        # 如果你的 DCT 系数生成时做了 / local_std，这里需要乘回去（强烈建议加上）
        trend_line = trend_line * local_std

        # === 3. 提取未来真实 OHLC 并减去 base_price ===
        base_price = X_raw[i, -1, 3]
        future_ohlc_raw = X_raw[i + 1: i + look_ahead + 1, -1, :4]
        future_ohlc_norm = future_ohlc_raw - base_price

        # === 4. 计算残差 Delta ===
        delta_ohlc = future_ohlc_norm - trend_line

        # === 5. 对 Delta 进行局部标准化（关键步骤）===
        delta_ohlc_normalized = delta_ohlc / local_std

        delta_list.append(delta_ohlc_normalized)

    delta_array = np.array(delta_list, dtype=np.float32)
    np.save(output_delta_file, delta_array)

    print(f">>> 残差数据制作完成: {output_delta_file}")
    print(f"    形状: {delta_array.shape}")
    print(f"    Delta 统计 -> mean: {delta_array.mean():.5f} | std: {delta_array.std():.5f}")


if __name__ == "__main__":
    make_delta_data()