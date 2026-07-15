"""
把训练好的 .pth 导出为 kline-de-pre 可直接加载的 ONNX。

严格匹配 app 的输入输出契约（从 src/utils/transformerService.js 逆推）：
  模型1 transformer_with_feat_merged.onnx:
    输入 : gaf_input  [N, 12, 60, 60]
    输出 : dct_output [N, 9]   (原始尺度 DCT 系数)
           feat_output[N, 128] (CNN 主干 GAP 特征, 供 Diffusion 条件用)

用法:
  python export_onnx.py <transformer_dct_v1.pth> <out.onnx>
  (默认: checkpoints/transformer_dct_v1.pth -> transformer_with_feat_merged.onnx)
"""
import os
import sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "CNN_Transformer"))
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


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "checkpoints", "transformer_dct_v1.pth")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "transformer_with_feat_merged.onnx")
    export_transformer(ckpt, out)
