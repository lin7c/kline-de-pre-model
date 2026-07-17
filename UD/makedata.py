import numpy as np
from scipy.fftpack import idct


ZIGZAG_THR = 0.5  # zigzag 反转阈值(局部 σ 单位; 路径已标准化, σ≈1)


def zigzag_pivot_mask(paths, thr=ZIGZAG_THR):
    """向量化 zigzag 枢轴检测(跨样本并行的状态机)。

    paths: (N, L) 标准化后的收盘路径。返回 (N, L) bool, 枢轴处为 True。
    起点与终点恒为枢轴(锚定两端), 中间为幅度 >= thr 的摆动折返点。
    """
    N, L = paths.shape
    mask = np.zeros((N, L), dtype=bool)
    mask[:, 0] = True
    direction = np.zeros(N, dtype=np.int8)      # 0 未定, 1 上行腿, -1 下行腿
    ext_val = paths[:, 0].copy()                 # 当前腿极值(未定方向时存最大值)
    ext_idx = np.zeros(N, dtype=np.int64)
    lo_val = paths[:, 0].copy()                  # 未定方向时同时跟踪最小值
    lo_idx = np.zeros(N, dtype=np.int64)

    rows = np.arange(N)
    for t in range(1, L):
        x = paths[:, t]

        und = direction == 0
        if und.any():
            hi_new = und & (x > ext_val)
            ext_val[hi_new], ext_idx[hi_new] = x[hi_new], t
            lo_new = und & (x < lo_val)
            lo_val[lo_new], lo_idx[lo_new] = x[lo_new], t
            up_start = und & (x - lo_val >= thr)     # 从最低点涨够 thr -> 上行腿
            if up_start.any():
                mask[rows[up_start], lo_idx[up_start]] = True
                direction[up_start] = 1
                ext_val[up_start], ext_idx[up_start] = x[up_start], t
            dn_start = und & (ext_val - x >= thr) & ~up_start
            if dn_start.any():
                mask[rows[dn_start], ext_idx[dn_start]] = True
                direction[dn_start] = -1
                ext_val[dn_start], ext_idx[dn_start] = x[dn_start], t

        up = direction == 1
        if up.any():
            better = up & (x >= ext_val)
            ext_val[better], ext_idx[better] = x[better], t
            rev = up & ~better & (ext_val - x >= thr)
            if rev.any():
                mask[rows[rev], ext_idx[rev]] = True   # 确认高点枢轴
                direction[rev] = -1
                ext_val[rev], ext_idx[rev] = x[rev], t

        dn = direction == -1
        if dn.any():
            better = dn & (x <= ext_val)
            ext_val[better], ext_idx[better] = x[better], t
            rev = dn & ~better & (x - ext_val >= thr)
            if rev.any():
                mask[rows[rev], ext_idx[rev]] = True   # 确认低点枢轴
                direction[rev] = 1
                ext_val[rev], ext_idx[rev] = x[rev], t

    # 收尾: 当前腿的极值 + 终点也作为枢轴
    mask[rows, ext_idx] = True
    mask[:, -1] = True
    return mask


def make_delta_data(input_x_file="../CNN/input_x_v1.npy",
                    dct_coeff_file="../CNN_Transformer/y_transformer_v1.npy",
                    output_delta_file="y_delta_ohlc.npy",
                    output_extras_file="ud_extras.npz"):
    """v3 变化:
    - DCT 系数 3 维(1m×3);
    - 趋势线重建后**锚点平移**(trend[0]=0), 幽灵K线起点=当前收盘(修 v2 前的起点漂移 bug);
    - 额外保存 fut_close / trend / pivots, 供 UD 训练的 zigzag 结构损失。
    """
    X_raw = np.load(input_x_file)
    dct_coeffs = np.load(dct_coeff_file)

    look_ahead = 60
    n = min(X_raw.shape[0], len(dct_coeffs)) - look_ahead
    print(f"剥离趋势提取残差(v3: 锚点平移 + zigzag 枢轴)... 有效样本 {n}")

    closes_end = X_raw[:, -1, 3].astype(np.float64)
    local_std = X_raw[:, :, 3].astype(np.float64).std(axis=1) + 1e-9

    idx = np.arange(n)
    k = np.arange(look_ahead)

    # 1) 趋势线: IDCT(3 系数补零到 60) + 锚点平移到 0
    full = np.zeros((n, look_ahead))
    full[:, :dct_coeffs.shape[1]] = dct_coeffs[:n]
    trend = idct(full, type=2, norm="ortho", axis=1)
    trend = trend - trend[:, :1]                      # ★ 锚点平移: trend[0]=0

    # 2) 未来真实 OHLC(标准化): (future - base)/σ
    fut_ohlc = X_raw[idx[:, None] + 1 + k[None, :], -1, :4].astype(np.float64)
    fut_norm = (fut_ohlc - closes_end[idx, None, None]) / local_std[idx, None, None]

    # 3) 残差 = 标准化未来 - 平移后趋势
    delta = (fut_norm - trend[:, :, None]).astype(np.float32)

    # 4) zigzag 枢轴(在标准化的未来收盘路径上)
    fut_close = fut_norm[:, :, 3].astype(np.float32)
    pivots = zigzag_pivot_mask(fut_close)

    np.save(output_delta_file, delta)
    np.savez_compressed(output_extras_file,
                        fut_close=fut_close,
                        trend=trend.astype(np.float32),
                        pivots=pivots)
    print(f">>> 残差: {output_delta_file} {delta.shape} | mean {delta.mean():.5f} std {delta.std():.5f}")
    print(f">>> extras: {output_extras_file} | 平均枢轴数/样本: {pivots.sum(1).mean():.2f}")


if __name__ == "__main__":
    make_delta_data()
