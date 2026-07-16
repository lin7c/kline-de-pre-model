"""
构建训练数据集（BTC 版）—— 直接从 OKX 拉取能拿到的最多历史 1m K 线。

数据源与 kline-de-pre 主项目一致：OKX BTC-USDT。训练一律在 Kaggle 等 GPU 环境
现拉现训，本地不再累积/保存任何训练数据。

多尺度窗口（v2，真·三时间尺度）：
  旧版把 5m/15m merge_asof 到 1m 时间轴，一个 60 分钟窗口里 5m 只有 12 个不同值、
  15m 只有 4 个——模型实际只看到 1 小时历史，却要预测未来 15 小时（标签 gap_15m=900），
  信息严重不对称。v2 改为每个样本包含三个「独立 60 根」窗口，右端对齐同一分钟 t：
    1m : t-59 .. t                （1 小时）
    5m : 60 个滚动 5 分钟块，块右端为 t, t-5, ..., t-295   （5 小时）
    15m: 60 个滚动 15 分钟块，块右端为 t, t-15, ..., t-885 （15 小时）
  「滚动块」= 以当前分钟为右端的 trailing 聚合（open=块首开、high/low=块内极值、close=当前收），
  只用 t 及更早的数据，无未来函数；推理端可从 900 根 1m 精确复现（见 src/utils/gafService.js）。

流程：
  从 OKX history-candles 分页拉取 1m（默认拉到历史尽头 = 能拿到的最多）
    -> 滚动聚合出 5m / 15m trailing 块
    -> 组装 (N, 60, 12) 多尺度窗口（每根 1m 一个样本，需 >=900 根历史）

输出：
  org_v1.csv  列: time, 1m_open/high/low/close, 5m_..., 15m_...（5m/15m 为右端在该分钟的滚动块）
  org_v1.npy  形状: (num_windows, 60, 12) float32，通道顺序 [1m OHLC, 5m OHLC, 15m OHLC]

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
LOOKBACK = 900          # 15m 窗口需要的最大 1m 回看（15*59+15）
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


def trailing_blocks(df_1m, period, prefix):
    """以每根 1m 为右端的 trailing `period` 分钟聚合块（只看当前及更早，无未来函数）。

    返回 (T, 4) float32: [open, high, low, close]，前 period-1 行为 NaN。
    """
    o = df_1m["1m_open"]
    h = df_1m["1m_high"].rolling(period).max()
    l = df_1m["1m_low"].rolling(period).min()
    blk = pd.DataFrame({
        f"{prefix}_open": o.shift(period - 1),   # 块首那根的开盘
        f"{prefix}_high": h,
        f"{prefix}_low": l,
        f"{prefix}_close": df_1m["1m_close"],    # 块右端 = 当前收盘
    })
    return blk


def build_multi_scale(df_1m):
    """组装逐分钟特征矩阵 F (T, 12) 与时间列：[1m OHLC, 5m块 OHLC, 15m块 OHLC]。"""
    print("🔄 计算 trailing 5m / 15m 滚动块...")
    blk5 = trailing_blocks(df_1m, 5, "5m")
    blk15 = trailing_blocks(df_1m, 15, "15m")
    combined = pd.concat([df_1m.reset_index(drop=True), blk5, blk15], axis=1)
    initial = len(combined)
    combined = combined.dropna().reset_index(drop=True)
    print(f"📊 聚合完成。原始 1m: {initial} 行 -> 有效: {len(combined)} 行")
    return combined


def create_multi_scale_windows(df, window_size=WINDOW_SIZE):
    """每个样本 = 右端对齐同一分钟 t 的三个独立 60 根窗口。

    行 j (0..59, 旧->新)：
      通道 0-3 : 1m  OHLC @ t-(59-j)
      通道 4-7 : 5m  滚动块 OHLC，块右端 @ t-(59-j)*5
      通道 8-11: 15m 滚动块 OHLC，块右端 @ t-(59-j)*15
    """
    vals = df.drop(columns=["time"]).values.astype(np.float32)  # (T, 12)
    T = len(vals)
    # 15m 窗口回看 15*(window_size-1)；df 已去掉 rolling 头部 NaN(14 行)，合计正好 LOOKBACK
    min_t = 15 * (window_size - 1)
    if T <= min_t:
        raise ValueError(f"有效数据量 {T} 不足以构成一个多尺度窗口（需 >{min_t}）")

    t_end = np.arange(min_t, T)                                  # 每个样本的右端索引
    back = (window_size - 1 - np.arange(window_size))            # 59..0
    X = np.empty((len(t_end), window_size, 12), dtype=np.float32)
    for g, stride in enumerate((1, 5, 15)):                      # 1m / 5m / 15m
        idx = t_end[:, None] - back[None, :] * stride            # (N, 60)
        X[:, :, g * 4:(g + 1) * 4] = vals[idx, g * 4:(g + 1) * 4]
    return X, len(t_end)


def run(inst_id="BTC-USDT", max_bars=None,
        output_file="org_v1.npy", csv_file="org_v1.csv"):
    df_1m = fetch_okx_1m(inst_id, max_bars)
    # 多尺度回看 900 + 标签前视 900，再留点余量
    if df_1m is None or len(df_1m) < LOOKBACK * 2 + 200:
        have = 0 if df_1m is None else len(df_1m)
        print(f"❌ 数据不足（当前 {have} 根，多尺度训练至少需 ~{LOOKBACK * 2 + 200} 根）。")
        sys.exit(1)

    full_df = build_multi_scale(df_1m)
    try:
        windows_data, actual_num = create_multi_scale_windows(full_df)
        np.save(output_file, np.ascontiguousarray(windows_data))
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
