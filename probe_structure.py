"""
结构探针实验：验证「高低点/走势结构」这类人类语义特征对未来方向是否有可预测性。

假设(用户提出): 判断 K 线只需要一两个表征(最高最低点、走势结构)。
若成立 → 用这些显式特征 + 最简单的分类器，就应能在诚实时序切分的样本外
把方向准确率做到明显 >50%(比如 53%+)。
若 ≈50% → 说明连人类语义的结构特征都不含前向信息，瓶颈是信号而非模型容量。

特征(每个样本在 60/300/900 分钟三个回看尺度上各算一组, 全部只用过去):
  pos   : 当前价在区间 [min,max] 的分位位置
  ret   : 区间收益 / 区间σ
  thi/tlo: 距区间最高/最低点过去了多久(归一化)
  dd/du : 距区间最高点回撤 / 距最低点反弹(σ 单位)
  hh/ll : 道氏结构近似(后半段高点是否抬高 / 低点是否降低)
  volr  : 短长波动率比 σ60/σ900

目标(未来 60/300/900 分钟, 与 DCT 标签同 horizon):
  dir : sign(close[t+W] - close[t])          终点方向
  c0  : sign(mean(close[t+1..t+W]) - close[t]) 均值方向(≈DCT c0 符号)

模型: LogisticRegression + HistGradientBoosting, 时序 80/20 切分留 900 gap。
用法: python probe_structure.py [max_bars=300000]
"""
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from getdata_btc import fetch_okx_1m
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

LOOK = 900          # 特征最大回看
HORIZONS = [60, 300, 900]
STRIDE = 5          # 每 5 分钟取一个样本(相邻窗口高度重叠, 采样即可)
GAP = 900           # 切分处留未来标签跨度的间隔


def build_features(c, ts):
    feats = {}
    sds = {}
    for W in (60, 300, 900):
        win = sliding_window_view(c, W)          # 视图: win[i] = c[i..i+W-1]
        rows = win[ts - W + 1].astype(np.float64)  # 每个样本以 t 为右端的窗口
        mx, mn = rows.max(1), rows.min(1)
        rng = mx - mn + 1e-9
        sd = rows.std(1) + 1e-9
        sds[W] = sd
        cur = c[ts]
        feats[f"pos_{W}"] = (cur - mn) / rng
        feats[f"ret_{W}"] = (cur - rows[:, 0]) / sd
        feats[f"thi_{W}"] = (W - 1 - rows.argmax(1)) / W
        feats[f"tlo_{W}"] = (W - 1 - rows.argmin(1)) / W
        feats[f"dd_{W}"] = (mx - cur) / sd
        feats[f"du_{W}"] = (cur - mn) / sd
        half = W // 2
        feats[f"hh_{W}"] = (rows[:, half:].max(1) > rows[:, :half].max(1)).astype(np.float64)
        feats[f"ll_{W}"] = (rows[:, half:].min(1) < rows[:, :half].min(1)).astype(np.float64)
    feats["volr"] = sds[60] / sds[900]
    names = list(feats)
    return np.column_stack([feats[k] for k in names]), names


def build_targets(c, ts):
    cs = np.concatenate([[0.0], np.cumsum(c, dtype=np.float64)])
    ys = {}
    for W in HORIZONS:
        fut_mean = (cs[ts + 1 + W] - cs[ts + 1]) / W
        ys[f"dir_{W}"] = (c[ts + W] > c[ts]).astype(int)
        ys[f"c0_{W}"] = (fut_mean > c[ts]).astype(int)
    return ys


def main():
    max_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    df = fetch_okx_1m("BTC-USDT", max_bars)
    c = df["1m_close"].values.astype(np.float64)
    T = len(c)
    print(f"1m 收盘序列: {T} 根")

    ts = np.arange(LOOK - 1, T - max(HORIZONS) - 1, STRIDE)
    X, names = build_features(c, ts)
    ys = build_targets(c, ts)
    print(f"样本: {len(ts)} | 特征: {len(names)} -> {names}")

    n = len(ts)
    ntr = int(n * 0.8)
    tr = np.arange(0, ntr)
    va = np.arange(min(ntr + GAP // STRIDE, n), n)
    print(f"时序切分 | 训练 {len(tr)} | gap {GAP // STRIDE} | 验证 {len(va)}")

    print(f"\n{'目标':<8} {'基线(多数类)':>12} {'LR acc':>8} {'LR auc':>8} {'GBDT acc':>9} {'GBDT auc':>9}")
    for key, y in ys.items():
        base = max(y[va].mean(), 1 - y[va].mean())
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        p_lr = lr.predict_proba(sc.transform(X[va]))[:, 1]
        gb = HistGradientBoostingClassifier(max_depth=4, max_iter=300,
                                            learning_rate=0.05, l2_regularization=1.0,
                                            random_state=42).fit(X[tr], y[tr])
        p_gb = gb.predict_proba(X[va])[:, 1]
        print(f"{key:<8} {base:>12.4f} "
              f"{accuracy_score(y[va], p_lr > 0.5):>8.4f} {roc_auc_score(y[va], p_lr):>8.4f} "
              f"{accuracy_score(y[va], p_gb > 0.5):>9.4f} {roc_auc_score(y[va], p_gb):>9.4f}")

    print("\n判读: acc 明显 >max(基线,0.5)+0.02 或 auc>0.53 → 结构特征含前向信息;"
          "\n      全部 ≈ 基线/0.5 → 结构特征无前向信息, 加大模型容量无意义。")


if __name__ == "__main__":
    main()
