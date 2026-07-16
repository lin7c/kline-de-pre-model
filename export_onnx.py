"""
把训练好的 .pth 导出为 kline-de-pre 可直接加载的 ONNX。

严格匹配 app 的输入输出契约（从 src/utils/transformerService.js / inferenceService.js 逆推）：
  模型1 transformer_with_feat_merged.onnx:
    输入 : gaf_input  [N, 12, 60, 60]
    输出 : dct_output [N, 9]   (原始尺度 DCT 系数)
           feat_output[N, 128] (CNN 主干 GAP 特征, 供 Diffusion 条件用)
  模型2 diffusion_unet_merged.onnx:
    输入 : x   [1, 60, 4] float32   (加噪残差)
           t   [1]        int64     (时间步)
           dct [1, 9]     float32
           feat[1, 128]   float32
    输出 : output_noise [1, 60, 4]  (预测噪声)

用法:
  python export_onnx.py <transformer_dct_v1.pth> <out.onnx>          # 只导模型1
  python export_onnx.py --diffusion <diffusion_delta_v1.pth> <out.onnx>  # 只导模型2
  (默认: checkpoints/transformer_dct_v1.pth -> transformer_with_feat_merged.onnx)
"""
import os
import sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "CNN_Transformer"))
sys.path.insert(0, os.path.join(HERE, "UD"))
from Dmodel import GafCnnTransformer


class TransformerExport(nn.Module):
    """包装成 (dct_output, feat_output) 双输出，匹配 app 契约。"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, gaf):
        feat = self.model.extract_feat(gaf)     # (B, 128)  = feat_output
        dct = self.model.regressor(feat)        # (B, 9)    = dct_output
        return dct, feat


def export_transformer(ckpt_path, out_path):
    model = GafCnnTransformer(output_dim=9)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    wrapper = TransformerExport(model).eval()
    dummy = torch.zeros(1, 12, 60, 60, dtype=torch.float32)  # CPU 导出，无需 GPU

    torch.onnx.export(
        wrapper, dummy, out_path,
        input_names=["gaf_input"],
        output_names=["dct_output", "feat_output"],
        dynamic_axes={
            "gaf_input": {0: "batch"},
            "dct_output": {0: "batch"},
            "feat_output": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,   # 用传统 TorchScript 导出器：可靠地保留 output_names 契约，且不依赖 onnxscript
    )
    print(f"✅ 已导出: {out_path}")

    # 自检输出维度
    with torch.no_grad():
        dct, feat = wrapper(dummy)
    print(f"   dct_output shape = {tuple(dct.shape)}  (应为 (1, 9))")
    print(f"   feat_output shape = {tuple(feat.shape)}  (应为 (1, 128))")


def export_diffusion(ckpt_path, out_path):
    from UDmodel import DiffusionUNet   # 延迟导入，避免只导模型1时的依赖

    model = DiffusionUNet(feature_dim=128, dct_dim=9, seq_len=60)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    dummy = (
        torch.zeros(1, 60, 4, dtype=torch.float32),   # x
        torch.zeros(1, dtype=torch.int64),            # t
        torch.zeros(1, 9, dtype=torch.float32),       # dct
        torch.zeros(1, 128, dtype=torch.float32),     # feat
    )
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["x", "t", "dct", "feat"],
        output_names=["output_noise"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,   # 传统导出器，稳定保留 input/output 名字契约
    )
    print(f"✅ 已导出: {out_path}")

    with torch.no_grad():
        out = model(*dummy)
    print(f"   output_noise shape = {tuple(out.shape)}  (应为 (1, 60, 4))")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--diffusion":
        ckpt = args[1] if len(args) > 1 else os.path.join(HERE, "checkpoints", "diffusion_delta_v1.pth")
        out = args[2] if len(args) > 2 else os.path.join(HERE, "diffusion_unet_merged.onnx")
        export_diffusion(ckpt, out)
    else:
        ckpt = args[0] if args else os.path.join(HERE, "checkpoints", "transformer_dct_v1.pth")
        out = args[1] if len(args) > 1 else os.path.join(HERE, "transformer_with_feat_merged.onnx")
        export_transformer(ckpt, out)
