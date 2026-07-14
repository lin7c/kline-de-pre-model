import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import sys
from UDmodel import DiffusionUNet, GaussianDiffusion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaf_transform import gasf_batch

def train_diffusion(MODEL_PATH, RESUME, X_FILE, DCT_FILE, DELTA_FILE, PRETRAINED_T_MODEL, epochs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    print(">>> 加载数据...")
    y_dct = np.load(DCT_FILE)      # 已经局部标准化后的 DCT 系数
    y_delta = np.load(DELTA_FILE)  # 已经局部标准化后的 Delta
    min_samples = len(y_delta)

    # 特征缓存：缓存名绑定原始窗口与特征提取模型的指纹(大小+修改时间)，
    # 数据或模型一变，缓存名随之改变，绝不会把旧特征套到新标签上。
    def _fp(p):
        s = os.stat(p)
        return f"{s.st_size}_{int(s.st_mtime)}"
    CACHE_FILE = f"X_feat_cache_{_fp(X_FILE)}_{_fp(PRETRAINED_T_MODEL)}.npy"
    if os.path.exists(CACHE_FILE):
        print(f">>> 加载缓存特征: {CACHE_FILE}")
        X_feat = np.load(CACHE_FILE)
    else:
        print(">>> 提取 CNN 特征（GASF 现算）...")
        X_raw = np.load(X_FILE)[:min_samples]
        from Dmodel import GafCnnTransformer
        t_model = GafCnnTransformer(output_dim=9).to(device)
        t_ckpt = torch.load(PRETRAINED_T_MODEL, map_location=device, weights_only=False)
        t_model.load_state_dict(t_ckpt.get('model_state_dict', t_ckpt))
        t_model.eval()

        all_feats = []
        with torch.no_grad():
            for i in range(0, len(X_raw), 128):
                batch = gasf_batch(torch.from_numpy(X_raw[i:i+128]).float().to(device))
                f = t_model.cnn(batch)
                f = torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).view(f.size(0), -1)
                all_feats.append(f.cpu().numpy())
        X_feat = np.concatenate(all_feats, axis=0)
        np.save(CACHE_FILE, X_feat)
        print(f">>> 特征提取完成并缓存: {CACHE_FILE}")
        del t_model

    X_feat = X_feat[:min_samples]
    y_dct = y_dct[:min_samples]
    y_delta = y_delta[:min_samples]

    print(f">>> 数据加载完成 | 样本数: {min_samples}")
    print(f">>> y_dct 统计: mean={y_dct.mean():.5f}, std={y_dct.std():.5f}")
    print(f">>> y_delta 统计: mean={y_delta.mean():.5f}, std={y_delta.std():.5f}")

    # === 不再做全局标准化，直接使用局部标准化后的数据 ===
    dataset = TensorDataset(
        torch.from_numpy(X_feat).float(),
        torch.from_numpy(y_dct).float(),
        torch.from_numpy(y_delta).float()
    )

    # 时序切分（同 CNN_Transformer）：按时间顺序切分并留 gap，防止重叠窗口/未来标签泄漏
    LABEL_GAP = 900  # 沿用管线中最大前视跨度
    train_split_idx = int(0.8 * min_samples)
    train_idx = list(range(0, train_split_idx))
    val_idx = list(range(min(train_split_idx + LABEL_GAP, min_samples), min_samples))
    if len(val_idx) == 0:
        raise ValueError(f"数据量太小，切分 gap({LABEL_GAP}) 后验证集为空，请累积更多数据")
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    print(f">>> 时序切分 | 训练: {len(train_idx)} | 间隔: {LABEL_GAP} | 验证: {len(val_idx)}")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # 模型与训练配置
    model = DiffusionUNet(feature_dim=X_feat.shape[1]).to(device)
    diffuser = GaussianDiffusion(timesteps=1000, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    start_epoch = 0

    if RESUME and os.path.exists(MODEL_PATH):
        print(f">>> 从 {MODEL_PATH} 恢复训练...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    early_stop_patience = 100
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for feats, dcts, deltas in train_loader:
            feats, dcts, deltas = feats.to(device), dcts.to(device), deltas.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, diffuser.timesteps, (feats.size(0),), device=device).long()
            noise = torch.randn_like(deltas)
            x_t = diffuser.sample_q_t(deltas, t, noise)

            pred_noise = model(x_t, t, dcts, feats)
            loss = criterion(pred_noise, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, dcts, deltas in val_loader:
                feats, dcts, deltas = feats.to(device), dcts.to(device), deltas.to(device)
                t = torch.randint(0, diffuser.timesteps, (feats.size(0),), device=device).long()
                noise = torch.randn_like(deltas)
                x_t = diffuser.sample_q_t(deltas, t, noise)
                pred_noise = model(x_t, t, dcts, feats)
                val_loss += criterion(pred_noise, noise).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1:03d}/{epochs}] | LR: {optimizer.param_groups[0]['lr']:.7f} | "
              f"Loss(T/V): {avg_train_loss:.6f}/{avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint_data, MODEL_PATH)
            print(f">>> 保存最佳模型 (Val Loss: {best_val_loss:.6f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("!!! 触发早停")
                break

    print(">>> 训练完成！")


if __name__ == "__main__":
    train_diffusion(
        MODEL_PATH="diffusion_delta_v1.pth",
        RESUME=False,
        X_FILE="../CNN/input_x_v1.npy",
        DCT_FILE="../CNN_Transformer/y_transformer_v1_g.npy",
        DELTA_FILE="y_delta_ohlc.npy",
        PRETRAINED_T_MODEL="../CNN_Transformer/transformer_dct_v1.pth"
    )