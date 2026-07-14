"""
训练流水线（精简版）——只训练两个模型：
  1. CNN_Transformer  (趋势/DCT 回归)  -> CNN_Transformer/transformer_dct_v1.pth
  2. UD               (Diffusion 残差)  -> UD/diffusion_delta_v1.pth

数据源：训练时直接从 OKX 拉取能拿到的最多 BTC-USDT 1m 历史（见 getdata_btc.py），
        在 Kaggle 等 GPU 环境现拉现训，本地不保存任何训练数据。
        强化学习（RL/PPO）部分已移除。

依赖链：
  OKX(现拉最多历史) -> getdata_btc.py -> org_v1.csv / org_v1.npy   (N,60,12)
  CNN/makedata.py           -> CNN/input_x_v1.npy                 (裁剪后的原始窗口 X)
  (GAF 不再预存，训练/推理时按 batch 现算，见 gaf_transform.py)
  CNN_Transformer/makedata  -> CNN_Transformer/y_transformer_v1.npy (DCT 标签)
  CNN_Transformer/train     -> transformer_dct_v1.pth            【模型1】
  CNN_Transformer/gy        -> y_transformer_v1_g.npy            (模型趋势输出，供 UD)
  UD/makedata               -> UD/y_delta_ohlc.npy               (残差标签)
  UD/train                  -> UD/diffusion_delta_v1.pth         【模型2】
"""
import subprocess
import os
import sys
import time


def run_script(script_path, description, args=None):
    abs_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(abs_path)
    script_name = os.path.basename(abs_path)

    print(f"\n--- 执行: {description} ---")
    cmd = [sys.executable, script_name] + (args or [])
    try:
        subprocess.run(cmd, cwd=script_dir, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"❌ 运行失败: {description}")
        return False


def main():
    # --- 阶段 1: 从 OKX 拉取最多历史并构建数据集 ---
    print("\n" + "=" * 28 + " 阶段 1: 拉取并构建 BTC 数据集 " + "=" * 28)
    if not run_script("getdata_btc.py", "从 OKX 拉取最多历史并构建数据集"):
        print("🚨 数据构建失败。")
        return

    # --- 阶段 2: 训练流水线 ---
    print("\n" + "=" * 28 + " 阶段 2: 训练流水线 " + "=" * 28)
    pipeline = [
        ("CNN/makedata.py",              "数据预处理 (原始窗口 input_x)"),
        ("CNN_Transformer/makedata.py",  "CNN-Transformer 标签 (DCT)"),
        ("CNN_Transformer/train.py",     "CNN-Transformer 训练【模型1】"),
        ("CNN_Transformer/gy.py",        "CNN-Transformer 推理 (UD 输入)"),
        ("UD/makedata.py",               "UD 残差预处理"),
        ("UD/train.py",                  "UD Diffusion 训练【模型2】"),
    ]

    total_start = time.time()
    for script, desc in pipeline:
        if not run_script(script, desc):
            print(f"🚨 流水线在「{desc}」中断。")
            return

    duration = (time.time() - total_start) / 60
    print(f"\n✅ 全部完成！耗时: {duration:.2f} 分钟")
    print("   模型1: CNN_Transformer/transformer_dct_v1.pth")
    print("   模型2: UD/diffusion_delta_v1.pth")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 已手动停止。")
