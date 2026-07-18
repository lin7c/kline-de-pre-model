import torch
import numpy as np
import os
import sys
from Dmodel import GafCnnTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaf_transform import gasf_batch
from struct_feat import struct_features


def run_inference(x_path="../CNN/input_x_v1.npy",
                  model_path="transformer_dct_v1.pth",
                  save_path="y_transformer_v1_g.npy"):
    """
    运行推理并直接保存模型的原始输出（不进行局部逆标准化）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return
    model = GafCnnTransformer.from_checkpoint(model_path, map_location=device).to(device)
    model.eval()
    print(f">>> 模型加载成功: {model_path}")

    # 2. 加载原始窗口（GASF 现算，不再读预存的 GAF 大文件）
    if not os.path.exists(x_path):
        print(f"错误：找不到输入数据 {x_path}")
        return

    X_raw = np.load(x_path)
    S = struct_features(X_raw)
    X_tensor = torch.from_numpy(X_raw).float()
    S_tensor = torch.from_numpy(S).float()

    # 3. 批量推理
    all_preds = []
    batch_size = 128
    print(f">>> 开始推理，总样本数: {len(X_tensor)}")

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_x = gasf_batch(X_tensor[i:i + batch_size].to(device))
            preds = model(batch_x, S_tensor[i:i + batch_size].to(device))
            all_preds.append(preds.cpu().numpy())

    # 4. 合并结果; v4 分位数模型(9维)时取 q50 段(3维)作为 UD 的 dct 条件
    y_final = np.concatenate(all_preds, axis=0)
    if y_final.shape[1] == 9:
        y_final = y_final[:, 3:6]
        print(">>> v4 分位数输出, 取 q50 段作为 UD 条件")

    # 保存
    np.save(save_path, y_final)
    print(f">>> 预测结果（标准化版本）已保存至: {save_path} | 形状: {y_final.shape}")
    print(f">>> 统计信息: mean={y_final.mean():.4f}, std={y_final.std():.4f}")


if __name__ == "__main__":
    run_inference()