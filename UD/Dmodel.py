import torch
import torch.nn as nn
import torch.nn.functional as F


class GafCnnTransformer(nn.Module):
    """
    类名保留以兼容既有 import。已移除原来的 Transformer：
      - 该 Transformer 直接把 CNN 的 15x15 空间特征拉平成 225 个 token 却没有位置编码，
        对空间顺序不敏感，近乎无效，只是白白增加参数、加剧过拟合。
    现结构：CNN 主干 -> 全局平均池化(128 维特征) -> 回归头(输出 9 维 DCT)。

    对外契约保持不变（kline-de-pre 依赖）：
      - 输入：GAF (B, 12, 60, 60)
      - 特征 feat：128 维（CNN 主干 GAP，供 Diffusion 条件用，= feat_output）
      - 输出 dct：9 维（= dct_output，原始尺度）
    """

    def __init__(self, input_channels=12, output_dim=9):
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

        # --- 回归头：加大 dropout 抑制过拟合（P0-2 正则）---
        self.regressor = nn.Sequential(
            nn.Linear(128, 128), nn.LeakyReLU(0.1), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        )

    def extract_feat(self, x):
        """CNN 主干 -> GAP -> 128 维特征（即 feat_output，Diffusion 的条件）。"""
        f = self.cnn(x)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)   # (B, 128)

    def forward(self, x):
        return self.regressor(self.extract_feat(x))     # (B, 9) = dct_output
