"""
Walk-forward 稳健性检查：确认结构特征的 60 分钟方向优势不是单一时段的运气。

方法: 取约 2 年 1m 数据, 样本按时间排序后, 前 40% 作为最小训练集,
其余切成 K=8 个连续验证折。第 k 折: 用折前全部数据训练(留 gap),
在该折上验证。报告每折的 acc/auc 与基线, 以及跨折均值±标准差。

判读: dir_60/c0_60 在大多数折(>=6/8)上 auc>0.52 且均值 auc>0.53 → 信号稳健;
      只有零星折达标 → 上次的优势是时段运气, 不值得改架构。

用法: python probe_walkforward.py [max_bars=1200000] [csv_path]
(在 Kaggle 上 csv 默认自动找 /kaggle/input 下最大的 CSV)
"""
import sys
import glob
import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

LOOK = 900
HORIZONS = [60, 300, 900]
STRIDE = 5
GAP = 900
K_FOLDS = 8


def build_features(c, ts):
    feats = {}
    sds = {}
    for W in (60, 300, 900):
        win = sliding_window_view(c, W)
        rows = win[ts - W + 1].astype(np.float32)
        mx, mn = rows.max(1), rows.min(1)
        rng = mx - mn + 1e-9
        sd = rows.std(1) + 1e-9
        sds[W] = sd
        cur = c[ts].astype(np.float32)
        feats[f"pos_{W}"] = (cur - mn) / rng
        feats[f"ret_{W}"] = (cur - rows[:, 0]) / sd
        feats[f"thi_{W}"] = (W - 1 - rows.argmax(1)) / W
        feats[f"tlo_{W}"] = (W - 1 - rows.argmin(1)) / W
        feats[f"dd_{W}"] = (mx - cur) / sd
        feats[f"du_{W}"] = (cur - mn) / sd
        half = W // 2
        feats[f"hh_{W}"] = (rows[:, half:].max(1) > rows[:, :half].max(1)).astype(np.float32)
        feats[f"ll_{W}"] = (rows[:, half:].min(1) < rows[:, :half].min(1)).astype(np.float32)
        del win, rows
    feats["volr"] = sds[60] / sds[900]
    names = list(feats)
    return np.column_stack([feats[k] for k in names]).astype(np.float32), names


def build_targets(c, ts):
    cs = np.concatenate([[0.0], np.cumsum(c, dtype=np.float64)])
    ys = {}
    for W in HORIZONS:
        fut_mean = (cs[ts + 1 + W] - cs[ts + 1]) / W
        ys[f"dir_{W}"] = (c[ts + W] > c[ts]).astype(int)
        ys[f"c0_{W}"] = (fut_mean > c[ts]).astype(int)
    return ys


def load_closes(max_bars, csv_path=None):
    if csv_path is None:
        cands = sorted(glob.glob("/kaggle/input/**/*.csv*", recursive=True),
                       key=lambda p: -os.path.getsize(p))
        csv_path = cands[0]
    print("数据文件:", csv_path)
    df = pd.read_csv(csv_path)
    cols = {k.lower(): k for k in df.columns}
    tcol = next(cols[k] for k in ("timestamp", "ts", "time", "date", "open time") if k in cols)
    df = df.sort_values(tcol).dropna(subset=[cols["close"]]).tail(max_bars)
    t = pd.to_datetime(df[tcol].values, unit="s", errors="coerce")
    print("时间范围:", t[0], "->", t[-1])
    return df[cols["close"]].values.astype(np.float64), t


def main():
    max_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 1200000
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    c, times = load_closes(max_bars, csv_path)
    T = len(c)
    ts = np.arange(LOOK - 1, T - max(HORIZONS) - 1, STRIDE)
    X, names = build_features(c, ts)
    ys = build_targets(c, ts)
    n = len(ts)
    print(f"样本: {n} | 特征: {len(names)}")

    start = int(n * 0.4)
    fold = (n - start) // K_FOLDS
    results = {k: [] for k in ys}

    for k in range(K_FOLDS):
        a, b = start + k * fold, start + (k + 1) * fold
        tr = np.arange(0, max(a - GAP // STRIDE, 1))
        va = np.arange(a, b)
        d0, d1 = times[ts[a]], times[ts[b - 1]]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        line = [f"折{k + 1} [{str(d0)[:10]}~{str(d1)[:10]}] n={len(va)}"]
        for key, y in ys.items():
            if len(np.unique(y[tr])) < 2:
                continue
            lr = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, y[tr])
            p = lr.predict_proba(Xva)[:, 1]
            base = max(y[va].mean(), 1 - y[va].mean())
            acc = accuracy_score(y[va], p > 0.5)
            auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else float("nan")
            results[key].append((base, acc, auc))
            if key in ("dir_60", "c0_60", "dir_300"):
                line.append(f"{key}: acc {acc:.4f}/基线 {base:.4f} auc {auc:.4f}")
        print(" | ".join(line), flush=True)

    print("\n==== 跨折汇总 (mean ± std) ====")
    print(f"{'目标':<8} {'基线':>14} {'acc':>16} {'auc':>16} {'auc>0.52 折数':>12}")
    for key, rs in results.items():
        rs = np.array(rs)
        good = int((rs[:, 2] > 0.52).sum())
        print(f"{key:<8} {rs[:, 0].mean():>7.4f}±{rs[:, 0].std():.4f} "
              f"{rs[:, 1].mean():>8.4f}±{rs[:, 1].std():.4f} "
              f"{rs[:, 2].mean():>8.4f}±{rs[:, 2].std():.4f} {good:>8}/{len(rs)}")

    print("\n判读: dir_60/c0_60 若 >=6/8 折 auc>0.52 且均值 auc>0.53 → 信号稳健, 按双通路架构开工;")
    print("      否则 → 上折优势属时段运气, 回到'编码器+情景推演'讨论。")


if __name__ == "__main__":
    main()
