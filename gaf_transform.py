"""
按 batch 现算 GASF —— 替代原来预存整个 (N, 12, 60, 60) 数组的做法。

原 CNN/GAF.py 把所有窗口的 GAF 图一次性摊进内存再存盘，每条长度 60 的序列
被展成 60×60，放大 60 倍、12 通道再 ×12：2 年 1m 数据 ≈ 86GB，必然爆内存/磁盘。

GASF 有闭式解，无需 arccos，可在 GPU 上按 batch 现算，峰值内存降到每 batch ~11MB：
    GASF[i,j] = s_i·s_j - sqrt(1-s_i^2)·sqrt(1-s_j^2)
其中 s 为每个窗口每个通道独立 min-max 归一化到 [-1,1] 的序列，
与原实现 (pyts summation, sample_range=(-1,1)) 数值完全一致（已用 pyts 校验 diff=0）。
"""
import torch


def gasf_batch(x):
    """
    x:      (B, L, C) 原始价格窗口（L=60 时间步, C=12 通道）
    return: (B, C, L, L) GASF 图，直接喂给 GafCnnTransformer
    """
    x = x.permute(0, 2, 1)                      # (B, C, L)
    mn = x.amin(dim=2, keepdim=True)
    mx = x.amax(dim=2, keepdim=True)
    rng = mx - mn
    s = ((x - mn) / rng.clamp_min(1e-9)) * 2 - 1
    # 常数序列（max==min）与原实现一致：归一化为 0
    s = torch.where(rng < 1e-9, torch.zeros_like(s), s)
    s = s.clamp(-1, 1)
    sq = torch.sqrt((1 - s * s).clamp_min(0))
    # GASF[b,c,i,j] = s_i s_j - sq_i sq_j
    gasf = s.unsqueeze(-1) * s.unsqueeze(-2) - sq.unsqueeze(-1) * sq.unsqueeze(-2)
    return gasf                                 # (B, C, L, L)
