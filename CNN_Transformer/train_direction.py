"""
方向头训练(v3.1)——冻结 v3 主干, 只训 Linear(153,1) 方向分类器。

背景: 直接回归 DCT 在弱信号下最优解=均值 → 趋势线永远趴平。
拆分: 方向(唯一有信号的维度, 探针 AUC 0.52-0.55)交给分类头;
      幅度/形状交给统计(类条件均值 c_up/c_dn), 推理时 dct = p·c_up + (1-p)·c_dn。

产物: direction_head_v1.pth = { head_state, c_up, c_dn, val_auc, val_acc }
模型2/JS/契约零改动(合成逻辑烘进 ONNX 导出包装层)。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from Dmodel import GafCnnTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaf_transform import gasf_batch
from struct_feat import struct_features


def extract_features(X, backbone, device, batch=256):
    """冻结主干提取 GAP128 特征(GASF 现算)。"""
    feats = []
    Xt = torch.from_numpy(X).float()
    with torch.no_grad():
        for i in range(0, len(Xt), batch):
            g = gasf_batch(Xt[i:i + batch].to(device))
            feats.append(backbone.extract_feat(g).cpu().numpy())
            if (i // batch) % 200 == 0:
                print(f"  特征提取 {i}/{len(Xt)}", flush=True)
    return np.concatenate(feats, axis=0)


def run(X_FILE="../CNN/input_x_v1.npy",
        Y_FILE="y_transformer_v1.npy",
        BACKBONE="transformer_dct_v1.pth",
        OUT="direction_head_v1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    X = np.load(X_FILE)
    y3 = np.load(Y_FILE)
    n = min(len(X), len(y3))
    X, y3 = X[:n], y3[:n]
    d_label = (y3[:, 0] > 0).astype(np.float32)          # 方向 = sign(c0)
    print(f">>> 样本 {n} | 上涨占比 {d_label.mean():.4f}")

    # 冻结主干
    backbone = GafCnnTransformer.from_checkpoint(BACKBONE, map_location=device).to(device)
    backbone.eval()

    S = struct_features(X)                                # (N,25)
    F = extract_features(X, backbone, device)             # (N,128)
    Z = np.concatenate([F, S], axis=1).astype(np.float32)  # (N,153)

    # 时序切分(与 train.py 一致)
    GAP = 900
    ntr = int(0.8 * n)
    tr = slice(0, ntr)
    va = slice(min(ntr + GAP, n), n)

    # 标准化(只用训练段统计)
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
    Zn = (Z - mu) / sd

    head = nn.Linear(Z.shape[1], 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()
    Ztr = torch.from_numpy(Zn[tr]).to(device)
    ytr = torch.from_numpy(d_label[tr]).to(device)
    Zva = torch.from_numpy(Zn[va]).to(device)
    yva = d_label[va]

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(200):
        head.train()
        perm = torch.randperm(len(Ztr), device=device)
        for i in range(0, len(perm), 4096):
            idx = perm[i:i + 4096]
            opt.zero_grad()
            loss = bce(head(Ztr[idx]).squeeze(1), ytr[idx])
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(Zva).squeeze(1)).cpu().numpy()
        auc = roc_auc_score(yva, p)
        acc = accuracy_score(yva, p > 0.5)
        print(f"Epoch {epoch + 1:03d} | val AUC {auc:.4f} | acc {acc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_acc = auc, acc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                print("早停")
                break

    # 类条件均值(只用训练段, 防泄漏)
    c_up = y3[tr][d_label[tr] > 0.5].mean(0)
    c_dn = y3[tr][d_label[tr] <= 0.5].mean(0)
    print(f">>> c_up = {np.round(c_up, 3)} | c_dn = {np.round(c_dn, 3)}")
    print(f">>> 最佳 val AUC {best_auc:.4f} | acc {best_acc:.4f} (探针参照: AUC 0.52-0.55)")

    torch.save({
        "head_state": best_state,
        "feat_mu": mu.astype(np.float32), "feat_sd": sd.astype(np.float32),
        "c_up": c_up.astype(np.float32), "c_dn": c_dn.astype(np.float32),
        "val_auc": best_auc, "val_acc": best_acc,
    }, OUT)
    print(f">>> 已保存 {OUT}")


if __name__ == "__main__":
    run()
