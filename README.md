# kline_de_pre (精简训练版)

让 AI 识别 K 线形态、预测未来走势。本目录为精简后的训练代码，**只训练两个模型**：

1. **CNN_Transformer** —— GAF 图像 → DCT 趋势系数回归（`transformer_dct_v1.pth`）
2. **UD** —— 条件 Diffusion，预测趋势之外的 OHLC 残差细节（`diffusion_delta_v1.pth`）

> 相比原始仓库：已移除强化学习(RL/PPO)；数据源由 MetaTrader5 换成 **OKX BTC-USDT**
> （与 kline-de-pre 主项目同源）；**GAF 改为按 batch 现算**，不再预存整张 `(N,12,60,60)`
> 大数组（否则 2 年数据 ≈ 86GB 必爆内存）。

## 训练（推荐 Kaggle GPU）

训练一律在 **Kaggle Notebook（GPU）** 上现拉现训，本地不保存任何训练数据。
直接用仓库里的 **`kaggle_train.ipynb`**：clone → 从 OKX 拉取能拿到的最多历史 → 训两个模型。

也可命令行：
```bash
pip install -r requirements.txt
python main.py          # 拉取 OKX 最多历史 → 预处理 → 训练两个模型
```

单独拉数据：
```bash
python getdata_btc.py            # 拉到历史尽头（最多，约 2 年）
python getdata_btc.py 400000     # 最多拉 40 万根 1m（约 9 个月）
```

## 目录结构

```
getdata_btc.py           从 OKX 拉 BTC-USDT 1m(最多历史)→重采样5m/15m→org_v1.csv/.npy (N,60,12)
gaf_transform.py         GASF 闭式解，按 batch 现算 (B,60,12)->(B,12,60,60)，替代预存 GAF
main.py                  训练流水线编排
CNN/
  makedata.py            org_v1.npy → input_x_v1.npy (裁剪对齐的原始窗口)
CNN_Transformer/
  Dmodel.py              GafCnnTransformer 网络
  makedata.py            生成 DCT 趋势标签 y_transformer_v1.npy
  train.py               训练【模型1】(时序切分, GASF 现算) → transformer_dct_v1.pth
  gy.py                  用【模型1】推理出趋势输出 y_transformer_v1_g.npy (供 UD)
UD/
  Dmodel.py / UDmodel.py DiffusionUNet + GaussianDiffusion
  makedata.py            生成残差标签 y_delta_ohlc.npy
  train.py               训练【模型2】(时序切分, 特征缓存带指纹) → diffusion_delta_v1.pth
  NK.py                  重建对比可视化（可选）
kaggle_train.ipynb       Kaggle 一键训练 notebook
```

## 依赖链

```
OKX(现拉最多历史) → getdata_btc.py → org_v1.{csv,npy}
   → CNN/makedata.py → CNN/input_x_v1.npy
   → CNN_Transformer/makedata.py → y_transformer_v1.npy
        → CNN_Transformer/train.py → transformer_dct_v1.pth   【模型1】(GASF 现算)
        → CNN_Transformer/gy.py    → y_transformer_v1_g.npy
   → UD/makedata.py → y_delta_ohlc.npy
        → UD/train.py → diffusion_delta_v1.pth                【模型2】(GASF 现算)
```

## 关键设计说明

- **GASF 现算**：`GASF[i,j] = s_i·s_j − √(1−s_i²)·√(1−s_j²)`（s 为每窗口每通道归一化到
  [-1,1] 的序列），与旧 pyts 实现数值一致（已校验 diff≈0），峰值内存从 86GB 降到每 batch ~11MB。
- **时序切分**：两个 train 按时间顺序切训练/验证集并留 900 根 gap，避免重叠窗口+未来标签泄漏
  （随机切分会让验证分数虚高失真）。
- **泛化**：想内化更长历史，就把 `getdata_btc.py` 拉更长的历史后从零重训——知识来自训练数据
  的跨度，不靠微调累加（微调会灾难性遗忘）。

作者：Lin7c (A2958358128@gmail.com)
