import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock1D(nn.Module):
    """ 1D 残差块，用于提取序列特征 """

    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # 用于将时间步和条件的嵌入注入到卷积层
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        # x: (B, C, L), emb: (B, emb_dim)
        h = self.conv1(F.silu(self.norm1(x)))

        # 注入嵌入信息
        emb_out = self.mlp(emb).unsqueeze(-1)
        h = h + emb_out

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class DiffusionUNet(nn.Module):
    def __init__(self, feature_dim=128, dct_dim=3, seq_len=60):  # v3: dct 3 维
        super().__init__()
        self.seq_len = seq_len

        # 1. 时间步的正弦位置编码
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        # 2. 条件融合 (CNN特征 + DCT)
        self.cond_mlp = nn.Sequential(
            nn.Linear(feature_dim + dct_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        # 3. UNet 结构 (Encoder -> Mid -> Decoder)
        # 输入维度: (B, 4, 60) -> 4 是 OHLC
        self.init_conv = nn.Conv1d(4, 64, kernel_size=3, padding=1)

        # Downsampling 部分 (这里通过残差块增加深度)
        self.down_block1 = ResidualBlock1D(64, 128, 512)  # 512 是 time + cond 的总维度
        self.down_block2 = ResidualBlock1D(128, 256, 512)

        # Middle
        self.mid_block1 = ResidualBlock1D(256, 256, 512)
        self.mid_block2 = ResidualBlock1D(256, 256, 512)

        # Upsampling 部分
        self.up_block1 = ResidualBlock1D(256 + 256, 128, 512)
        self.up_block2 = ResidualBlock1D(128 + 128, 64, 512)

        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv1d(64, 4, kernel_size=1)

    def get_time_embedding(self, timesteps):
        """ 生成正弦时间嵌入 """
        half_dim = 64
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def forward(self, x, t, dcts, feats):
        """
        x: (B, 60, 4) - 加噪后的 Delta
        t: (B,) - 时间步
        dcts: (B, 9)
        feats: (B, feature_dim)
        """
        # 转为 (B, 4, 60) 适配 Conv1d
        x = x.transpose(1, 2)

        # 获取嵌入并拼接
        t_emb = self.time_mlp(self.get_time_embedding(t))  # (B, 256)
        c_emb = self.cond_mlp(torch.cat([dcts, feats], dim=1))  # (B, 256)
        emb = torch.cat([t_emb, c_emb], dim=1)  # (B, 512)

        # UNet 前向
        x1 = self.init_conv(x)  # (B, 64, 60)
        x2 = self.down_block1(x1, emb)  # (B, 128, 60)
        x3 = self.down_block2(x2, emb)  # (B, 256, 60)

        m = self.mid_block1(x3, emb)
        m = self.mid_block2(m, emb)

        u1 = self.up_block1(torch.cat([m, x3], dim=1), emb)  # (B, 128, 60)
        u2 = self.up_block2(torch.cat([u1, x2], dim=1), emb)  # (B, 64, 60)

        out = self.out_conv(F.silu(self.out_norm(u2)))
        return out.transpose(1, 2)  # 还原回 (B, 60, 4)


class GaussianDiffusion:
    """ 负责加噪逻辑的工具类 """

    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 前向加噪所需系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_q_t(self, x_0, t, noise):
        """ 得到加噪后的 x_t """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise