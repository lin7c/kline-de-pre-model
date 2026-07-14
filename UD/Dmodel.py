import torch
import torch.nn as nn

class GafCnnTransformer(nn.Module):
    def __init__(self, input_channels=12, output_dim=3):
        super(GafCnnTransformer, self).__init__()

        # --- 1. CNN 骨干：提取空间形态特征 ---
        # 注意：PyTorch Conv2d 内部计算依然遵循 (Channels, Height, Width)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),  # 60x60 -> 30x30

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2),  # 30x30 -> 15x15

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)  # 输出特征图: (Batch, 128, 15, 15)
        )

        # --- 2. Transformer 部分 ---
        self.d_model = 128
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- 3. 回归头 (Regression Head) ---
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim)
            # 移除 Sigmoid，适配连续趋势得分 (Trend Score)
        )

    def forward(self, x):
        x = self.cnn(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.d_model, -1).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.gap(x).squeeze(-1)
        return self.regressor(x)