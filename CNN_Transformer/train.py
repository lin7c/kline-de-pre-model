import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import sys
from Dmodel import GafCnnTransformer
from sklearn.metrics import r2_score

# GASF 按 batch 现算（不再预存 (N,12,60,60) 大数组，避免内存/磁盘爆炸）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gaf_transform import gasf_batch
from struct_feat import struct_features


def train_model(MODEL_PATH, RESUME, X_FILE, Y_FILE, output_dim=9, epochs=10000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # 1. 加载数据
    if not os.path.exists(X_FILE) or not os.path.exists(Y_FILE):
        raise FileNotFoundError(f"找不到数据文件: {X_FILE} 或 {Y_FILE}")

    X = np.load(X_FILE)
    y = np.load(Y_FILE)

    min_samples = min(len(X), len(y))
    X = X[:min_samples]
    y = y[:min_samples]

    print(f">>> 数据加载完成 | 样本数: {min_samples}")
    print(f">>> X(原始窗口) shape: {X.shape} | Y shape: {y.shape}  (GASF 训练时按 batch 现算)")
    print(
        f">>> Y 统计信息 (应接近 mean≈0, std≈1): mean={np.mean(y):.4f}, std={np.std(y):.4f}, min={np.min(y):.4f}, max={np.max(y):.4f}")

    # 结构特征旁路(v3): 从窗口显式计算 25 维结构表征
    S = struct_features(X)
    print(f">>> 结构特征: {S.shape}")

    # 直接使用 makedata.py 生成的标准化后的 Y
    X_tensor = torch.from_numpy(X).float()
    S_tensor = torch.from_numpy(S).float()
    y_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(X_tensor, S_tensor, y_tensor)
    # 时间序列必须按时间顺序切分：样本是重叠滑动窗口且带未来标签，
    # 随机切分会把相邻近乎相同的窗口分到 train/val，造成数据泄漏、验证指标虚高。
    # 切分处再留 LABEL_GAP 个样本的间隔，断开标签向后看造成的重叠。
    LABEL_GAP = 900  # 保守间隔: 输入回看 900(15m×60), 标签前视 60
    n = len(dataset)
    train_size = int(0.8 * n)
    train_idx = list(range(0, train_size))
    val_idx = list(range(min(train_size + LABEL_GAP, n), n))
    if len(val_idx) == 0:
        raise ValueError(f"数据量太小，切分 gap({LABEL_GAP}) 后验证集为空，请累积更多数据")
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    print(f">>> 时序切分 | 训练: {len(train_idx)} | 间隔: {LABEL_GAP} | 验证: {len(val_idx)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # 2. 模型、优化器、损失
    model = GafCnnTransformer(output_dim=output_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-3)  # 加大 weight_decay 抑制过拟合(P0-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    criterion = nn.HuberLoss(delta=1.0)  # 在标准化空间建议使用 1.0 左右

    best_val_loss = float('inf')
    start_epoch = 0

    # 3. Resume 逻辑（简化）
    if RESUME and os.path.exists(MODEL_PATH):
        print(f">>> 从 {MODEL_PATH} 恢复训练...")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f">>> 已恢复，最佳 val_loss: {best_val_loss:.5f}")
        else:
            model.load_state_dict(checkpoint)

    # 4. 训练循环
    early_stop_patience = 100
    epochs_no_improve = 0

    for epoch in range(start_epoch, epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for inputs, sfeat, labels in train_loader:
            inputs, sfeat, labels = inputs.to(device), sfeat.to(device), labels.to(device)
            inputs = gasf_batch(inputs)  # (B,60,12) -> (B,12,60,60) 现算
            optimizer.zero_grad()
            outputs = model(inputs, sfeat)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, sfeat, labels in val_loader:
                inputs, sfeat, labels = inputs.to(device), sfeat.to(device), labels.to(device)
                inputs = gasf_batch(inputs)
                outputs = model(inputs, sfeat)
                val_loss += criterion(outputs, labels).item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 在标准化后的 DCT 空间计算 R²（干净且合理）
        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        current_r2 = r2_score(y_true, y_pred)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch + 1:03d}/{epochs}] | LR: {optimizer.param_groups[0]['lr']:.7f} | "
              f"Loss(T/V): {avg_train_loss:.5f}/{avg_val_loss:.5f} | R²: {current_r2:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint_data, MODEL_PATH)
            print(f">>> 保存最佳模型 (val_loss: {best_val_loss:.5f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"!!! 早停触发（连续 {early_stop_patience} 个 epoch 无改善）")
                break
    for i in range(y.shape[1]):
        col_mean = np.mean(y[:, i])
        col_std = np.std(y[:, i])
        col_max = np.max(y[:, i])
        col_min = np.min(y[:, i])
        print(f"维度 {i:02d} | 均值: {col_mean:8.4f} | 标准差: {col_std:8.4f} | 范围: [{col_min:8.2f}, {col_max:8.2f}]")
    print(">>> 训练完成！")
    return model


def run():
    CONFIG = {
        "MODEL_PATH": "transformer_dct_v1.pth",
        "RESUME": False,  # 换数据集或想重新训练时设为 False；这里默认从零训练 BTC
        "X_FILE": "../CNN/input_x_v1.npy",  # 原始窗口(N,60,12)，GASF 训练时现算
        "Y_FILE": "y_transformer_v1.npy",
        "output_dim": 3,
        "epochs": int(os.environ.get("CT_EPOCHS", 10000))  # 可用环境变量限制轮次(测试用)
    }
    train_model(**CONFIG)


if __name__ == "__main__":
    run()