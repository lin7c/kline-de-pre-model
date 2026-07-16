import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from getdata_btc import build_multi_scale, create_multi_scale_windows

# 合成 1m: close = 索引值 i, open=i-0.5, high=i+1, low=i-1
T = 2000
i = np.arange(T, dtype=float)
df = pd.DataFrame({
    "time": pd.date_range("2026-01-01", periods=T, freq="1min"),
    "1m_open": i - 0.5, "1m_high": i + 1.0, "1m_low": i - 1.0, "1m_close": i,
})
full = build_multi_scale(df)
X, N = create_multi_scale_windows(full)
print("X shape:", X.shape, "| N:", N)

# full 去掉了前 14 行 NaN，所以 full 行 k 对应原始分钟 k+14
# 样本 s 的右端 t = (s + 885) 行 -> 原始分钟 m_t = s + 885 + 14 = s + 899
s = 300
m_t = s + 899
# 1m close 通道 (col 3): 应为 m_t-59 .. m_t
assert np.allclose(X[s, :, 3], np.arange(m_t - 59, m_t + 1)), X[s, -5:, 3]
# 5m close 通道 (col 7): 块右端 m_t-295, m_t-290, ..., m_t
assert np.allclose(X[s, :, 7], m_t - (59 - np.arange(60)) * 5), X[s, -5:, 7]
# 15m close (col 11): m_t-885 .. m_t 步长 15
assert np.allclose(X[s, :, 11], m_t - (59 - np.arange(60)) * 15)
# 5m open (col 4): 块首开盘 = (右端-4) 的 open = 右端-4-0.5
assert np.allclose(X[s, :, 4], m_t - (59 - np.arange(60)) * 5 - 4 - 0.5)
# 5m high (col 5): 块内最大 high = 右端 high = 右端+1
assert np.allclose(X[s, :, 5], m_t - (59 - np.arange(60)) * 5 + 1)
# 15m low (col 10): 块内最小 low = 块首 low = 右端-14-1
assert np.allclose(X[s, :, 10], m_t - (59 - np.arange(60)) * 15 - 15)
# 无未来函数: 样本 s 所有值 <= m_t + 1 (high 加了1)
assert X[s].max() <= m_t + 1
# 首样本回看正好用满: 样本 0 的 15m 窗口最旧块首 = 分钟 0
assert X[0, 0, 10] == 0 - 1  # low of minute 0 = -1
print("✅ 多尺度窗口全部断言通过")

# 标签兼容性: makedata 期望 X[i+k,-1,7] = 未来第 k 分钟收盘
k = 37
assert X[s + k, -1, 7] == m_t + k
assert X[s + k, -1, 11] == m_t + k
print("✅ makedata 标签语义兼容 (X[i+k,-1,close] = 未来第 k 分钟收盘)")
