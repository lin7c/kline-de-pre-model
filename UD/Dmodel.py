import torch
import torch.nn as nn
import torch.nn.functional as F


class GafCnnTransformer(nn.Module):
    """
    类名保留以兼容既有 import。v3 双通路结构：

      GAF (B,12,60,60) ──CNN──> GAP feat(128) ┐
                                              ├─ concat ─> 单线性层 ─> dct (B,3)
      结构特征 sfeat (B,25) ──────────────────┘

    设计依据(probe_structure / probe_walkforward 实验):
      - 显式结构特征(高低点位置/回撤/道氏结构)含微弱但从不为负的前向信息,
        GAF-CNN 无法自行从纹理中提取 → 旁路直连回归头;
      - 回归头退化为单线性层: v8 训练曲线证明多层 MLP 头只会加速背题
        (train loss 一路降、val loss 一路升, 最佳仅在 epoch 2)。

    对外契约(kline-de-pre v3):
      - 输入: GAF (B,12,60,60) + 结构特征 (B,25)
      - feat_output: 128 维(CNN GAP, 供 Diffusion 条件用)
      - dct_output : 3 维(1m 未来 60 分钟趋势的前 3 个 DCT 系数, 局部标准化空间)
    """

    def __init__(self, input_channels=12, output_dim=3, struct_dim=25):
        super().__init__()

        # --- CNN 主干：末层固定 128 通道 -> GAP 后即 128 维 feat_output（契约要求）---
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.MaxPool2d(2),   # 60->30
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),   # 30->15
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),                    # (B,128,15,15)
        )
        self.feat_dim = 128
        self.struct_dim = struct_dim

        # --- 回归头：单线性层(CNN feat + 结构特征旁路) ---
        self.regressor = nn.Linear(self.feat_dim + struct_dim, output_dim)

    def extract_feat(self, x):
        """CNN 主干 -> GAP -> 128 维特征（即 feat_output，Diffusion 的条件）。"""
        f = self.cnn(x)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)   # (B, 128)

    def forward(self, x, sfeat):
        """x: GAF (B,12,60,60); sfeat: 结构特征 (B,25) -> dct (B,3)"""
        return self.regressor(torch.cat([self.extract_feat(x), sfeat], dim=1))
