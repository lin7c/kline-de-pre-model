"""
结构特征(25 维)——从 v2 多尺度窗口 (N,60,12) 直接计算,不引入任何新数据。

探针实验(probe_structure/probe_walkforward)证明这些显式表征含有微弱但
从不为负的前向信息(dir_60 AUC 8/8 折 >0.5, 近一年 0.53-0.55),
而 GAF-CNN 无法自行从纹理中提取它们 → 作为旁路直接喂给线性回归头。

三个尺度分别用窗口的 1m/5m/15m 通道组(各 60 根, 覆盖 1h/5h/15h):
  pos  : 现价在区间 [min(low), max(high)] 的分位位置
  ret  : (现收 - 首开) / σ(closes)
  thi/tlo: 最高/最低点出现在窗口内多久之前(归一化)
  dd/du: 距区间最高点回撤 / 距最低点反弹(σ 单位)
  hh/ll: 道氏结构(后半段高点是否抬高 / 低点是否降低)
  volr : σ(1m closes)/σ(15m closes)

与 JS 端 src/utils/gafService.js 的 structFeatures() 必须保持一致。
"""
import numpy as np

FEAT_DIM = 25
_GROUPS = ((0, "1m"), (4, "5m"), (8, "15m"))  # (通道组起始列, 名称); 列序 O,H,L,C


def struct_features(X):
    """X: (N, 60, 12) float -> (N, 25) float32。全部只用窗口内(过去)数据。"""
    N, W, _ = X.shape
    feats = []
    sds = {}
    for base, name in _GROUPS:
        o = X[:, :, base + 0].astype(np.float64)
        h = X[:, :, base + 1].astype(np.float64)
        l = X[:, :, base + 2].astype(np.float64)
        c = X[:, :, base + 3].astype(np.float64)
        cur = c[:, -1]
        mx, mn = h.max(1), l.min(1)
        rng = mx - mn + 1e-9
        sd = c.std(1) + 1e-9
        sds[name] = sd
        half = W // 2
        feats.extend([
            (cur - mn) / rng,                          # pos
            (cur - o[:, 0]) / sd,                      # ret
            (W - 1 - h.argmax(1)) / W,                 # thi
            (W - 1 - l.argmin(1)) / W,                 # tlo
            (mx - cur) / sd,                           # dd
            (cur - mn) / sd,                           # du
            (h[:, half:].max(1) > h[:, :half].max(1)).astype(np.float64),  # hh
            (l[:, half:].min(1) < l[:, :half].min(1)).astype(np.float64),  # ll
        ])
    feats.append(sds["1m"] / sds["15m"])               # volr
    out = np.column_stack(feats).astype(np.float32)
    assert out.shape == (N, FEAT_DIM)
    return out


if __name__ == "__main__":
    # 跨语言一致性自检: 与 JS 端 structFeatures() 在同一确定性窗口上的输出对拍。
    # 生成公式两端一致: v[j,k] = sin(j*0.37 + k*1.3)*10 + j*0.05*(k%4) + 100
    j = np.arange(60)[:, None]
    k = np.arange(12)[None, :]
    X = (np.sin(j * 0.37 + k * 1.3) * 10 + j * 0.05 * (k % 4) + 100)[None, :, :]
    F = struct_features(X)
    assert np.isfinite(F).all()
    JS_EXPECTED = np.array([
        1.088944, 1.914758, 0.116667, 0.883333, -0.257343, 3.150646, 1, 0,
        0.609139, 1.753041, 0.066667, 0.833333, 1.140007, 1.776646, 1, 0,
        0.351461, 0.955573, 0.016667, 0.8, 1.863379, 1.009814, 1, 0,
        1.013321], dtype=np.float64)
    diff = np.abs(F[0].astype(np.float64) - JS_EXPECTED).max()
    assert diff < 1e-4, f"与 JS 端不一致, max diff = {diff}"
    print(f"OK {F.shape} | 与 JS structFeatures 最大差异 {diff:.2e}")
