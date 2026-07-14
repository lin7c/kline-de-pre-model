import numpy as np


def generate_y_from_npy(file_path="btcusdm_multi_tf_windows.npy"):
    # 1. 加载 X 原始窗口数据
    X_raw = np.load(file_path)
    total_samples = X_raw.shape[0]

    # 定义未来需要的跨度
    gap_1m = 60
    gap_5m = 300
    gap_15m = 900

    valid_samples = total_samples - gap_15m

    if valid_samples <= 0:
        print("数据量太小，不足以支撑标签生成")
        return None, None

    y_list = []

    print(f"正在从滑窗提取标签（已加入局部标准化）... 有效样本数: {valid_samples}")

    for i in range(valid_samples):
        # --- 核心新增：计算当前窗口的历史波动率作为局部标准差 ---
        # 使用 1m 周期（索引 3 是 close）的当前 60 根 K 线计算
        local_std = np.std(X_raw[i, :, 3]) + 1e-9

        # 当前窗口的收盘价
        current_close = X_raw[i, -1, 3]

        # 提取未来序列池
        future_data_pool = X_raw[i + 1: i + gap_15m + 1, -1, :]

        # 1. 未来 1m (60分钟) 的原始变动量 (未来价 - 当前价)
        fut_1m_high_delta = np.max(future_data_pool[:gap_1m, 1]) - current_close
        fut_1m_low_delta = np.min(future_data_pool[:gap_1m, 2]) - current_close

        # 2. 未来 5m (300分钟) 的原始变动量
        fut_5m_high_delta = np.max(future_data_pool[:gap_5m, 5]) - current_close
        fut_5m_low_delta = np.min(future_data_pool[:gap_5m, 6]) - current_close

        # 3. 未来 15m (900分钟) 的原始变动量
        fut_15m_high_delta = np.max(future_data_pool[:gap_15m, 9]) - current_close
        fut_15m_low_delta = np.min(future_data_pool[:gap_15m, 10]) - current_close

        # --- 执行局部标准化 ---
        # 逻辑：(未来变动量 / 历史波动标准差)
        # 这反映了未来价格穿透了多少个历史标准差单位
        y_list.append([
            fut_1m_high_delta / local_std, fut_1m_low_delta / local_std,
            fut_5m_high_delta / local_std, fut_5m_low_delta / local_std,
            fut_15m_high_delta / local_std, fut_15m_low_delta / local_std
        ])

    X_final = X_raw[:valid_samples]
    y_final = np.array(y_list, dtype=np.float32)

    return X_final, y_final


def run(input_file="../org_v1.npy", X_FILE="input_x_v1.npy", Y_FILE="y_cnn_v1.npy"):
    X, y = generate_y_from_npy(input_file)

    if X is not None:
        print(f"处理完成！")
        print(f"y 统计 (局部标准化后) -> mean: {y.mean():.4f} | std: {y.std():.4f}")
        print(f"y 范围 -> min: {y.min():.2f} | max: {y.max():.2f}")

        np.save(X_FILE, X)
        np.save(Y_FILE, y)


if __name__ == "__main__":
    run()