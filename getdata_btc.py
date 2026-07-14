"""
构建训练数据集（BTC 版）—— 直接从 OKX 拉取能拿到的最多历史 1m K 线。

数据源与 kline-de-pre 主项目一致：OKX BTC-USDT。训练一律在 Kaggle 等 GPU 环境
现拉现训，本地不再累积/保存任何训练数据。

流程：
  从 OKX history-candles 分页拉取 1m（默认拉到历史尽头 = 能拿到的最多）
    -> 重采样出 5m / 15m（左边界，与原 MT5 bar 开盘时间语义一致）
    -> merge_asof(backward) 对齐到 1m 时间轴（防未来函数）
    -> (N, 60, 12) 滑动窗口（步长固定为 1，保证下游未来标签的分钟语义正确）

输出（与原实现一致）：
  org_v1.csv  列: time,1m_open/high/low/close,5m_...,15m_...
  org_v1.npy  形状: (num_windows, 60, 12) float32

用法:
  python getdata_btc.py                # 拉到历史尽头（最多）
  python getdata_btc.py 200000         # 最多拉 20 万根 1m
  python getdata_btc.py all BTC-USDT   # 显式拉满 + 指定品种
"""
import sys
import time
import numpy as np
import pandas as pd
import requests

WINDOW_SIZE = 60
OKX_BASE = "https://www.okx.com"
PAGE_LIMIT = 100  # OKX history-candles 单次上限


def fetch_okx_1m(inst_id, max_bars):
    """从 OKX 分页拉取 1m。max_bars=None 表示拉到历史尽头。返回按时间升序的 DataFrame。"""
    tgt = "历史尽头(最多)" if max_bars is None else f"{max_bars} 根"
    print(f"📡 从 OKX 拉取 {inst_id} 的 1m K 线（目标: {tgt}）...")
    rows = []
    after = ""       # 分页游标：返回 ts < after 的更早数据
    session = requests.Session()
    t0 = time.time()

    while max_bars is None or len(rows) < max_bars:
        params = {"instId": inst_id, "bar": "1m", "limit": str(PAGE_LIMIT)}
        if after:
            params["after"] = after
        try:
            resp = session.get(f"{OKX_BASE}/api/v5/market/history-candles",
                               params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as e:
            print(f"\n⚠️ 请求失败，重试中: {e}")
            time.sleep(1)
            continue

        if payload.get("code") != "0":
            print(f"\n❌ OKX 返回错误: {payload.get('msg')}")
            break

        data = payload.get("data", [])
        if not data:
            print("\nℹ️ 已到历史数据尽头。")
            break

        rows.extend(data)
        after = data[-1][0]  # 最旧一条的时间戳作为下一页游标
        if len(rows) % 5000 < PAGE_LIMIT:
            rate = len(rows) / max(time.time() - t0, 1e-9)
            print(f"  已获取 {len(rows)} 根 (~{rate:.0f}/s)...", end="\r")
        time.sleep(0.12)  # 限速，避免触发 OKX 频控

    print(f"\n✅ 拉取完成: {len(rows)} 根，耗时 {(time.time()-t0)/60:.1f} 分钟")
    if not rows:
        return None
    if max_bars is not None:
        rows = rows[:max_bars]

    # OKX candle: [ts_ms, open, high, low, close, vol, ...]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "vol", "volCcy", "volCcyQuote", "confirm"])
    df = df[["ts", "open", "high", "low", "close"]].copy()
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df["time"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    df = df.drop(columns=["ts"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    df = df[["time", "open", "high", "low", "close"]]
    df.columns = ["time"] + [f"1m_{c}" for c in ["open", "high", "low", "close"]]
    return df


def resample_tf(df_1m, rule, prefix):
    s = df_1m.set_index("time")
    agg = s.resample(rule, closed="left", label="left").agg(
        **{
            f"{prefix}_open": ("1m_open", "first"),
            f"{prefix}_high": ("1m_high", "max"),
            f"{prefix}_low": ("1m_low", "min"),
            f"{prefix}_close": ("1m_close", "last"),
        }
    ).dropna().reset_index()
    return agg


def build_aligned(df_1m):
    print("🔄 重采样 5m / 15m 并跨周期对齐...")
    df_5m = resample_tf(df_1m, "5min", "5m")
    df_15m = resample_tf(df_1m, "15min", "15m")
    combined = pd.merge_asof(df_1m, df_5m, on="time", direction="backward")
    combined = pd.merge_asof(combined, df_15m, on="time", direction="backward")
    initial = len(combined)
    combined.dropna(inplace=True)
    print(f"📊 对齐完成。原始 1m: {initial} 行 -> 有效对齐: {len(combined)} 行")
    return combined.reset_index(drop=True)


def create_sliding_windows(df, window_size):
    data_values = df.drop(columns=["time"]).values.astype(np.float32)
    total_len = len(data_values)
    if total_len < window_size:
        raise ValueError(f"有效数据量 {total_len} 小于窗口大小 {window_size}")
    num_windows = total_len - window_size + 1
    shape = (num_windows, window_size, data_values.shape[1])
    strides = (data_values.strides[0], data_values.strides[0], data_values.strides[1])
    windows = np.lib.stride_tricks.as_strided(data_values, shape=shape, strides=strides)
    return windows, num_windows


def run(inst_id="BTC-USDT", max_bars=None,
        output_file="org_v1.npy", csv_file="org_v1.csv"):
    df_1m = fetch_okx_1m(inst_id, max_bars)
    if df_1m is None or len(df_1m) < WINDOW_SIZE + 1000:
        have = 0 if df_1m is None else len(df_1m)
        print(f"❌ 数据不足（当前 {have} 根，训练至少需 ~1100 根）。")
        sys.exit(1)

    full_df = build_aligned(df_1m)
    try:
        windows_data, actual_num = create_sliding_windows(full_df, WINDOW_SIZE)
        np.save(output_file, np.ascontiguousarray(windows_data))  # as_strided 是视图，需转连续
        full_df.to_csv(csv_file, index=False)
        print("-" * 30)
        print("✅ 数据集构建完成！")
        print(f"📂 导出: {output_file} & {csv_file}")
        print(f"📐 形状: {windows_data.shape} (样本数, 窗口长度, 特征数) float32")
        print(f"🕒 时间范围: {full_df['time'].iloc[0]} 至 {full_df['time'].iloc[-1]}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ 处理窗口时出错: {e}")
        sys.exit(1)


def _parse_args(argv):
    max_bars = None      # 默认拉满
    inst_id = "BTC-USDT"
    if len(argv) > 1 and argv[1].lower() not in ("all", "max", "full"):
        max_bars = int(argv[1])
    if len(argv) > 2:
        inst_id = argv[2]
    return inst_id, max_bars


if __name__ == "__main__":
    inst, mb = _parse_args(sys.argv)
    run(inst_id=inst, max_bars=mb)
