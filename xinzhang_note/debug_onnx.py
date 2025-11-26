import torch
import torch.nn as nn

class ToyAlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 模拟几个简单的层
        self.proj_vision = nn.Linear(1024, 4096)
        self.proj_text = nn.Linear(4096, 4096)

    def forward(self, text_embeds, image_features):
        # 模拟 VLA 的拼接逻辑
        # A. 假设 image_features 输入是 [B, 576, 1024]
        # 将其投影对齐到 LLM 维度
        img_proj = self.proj_vision(image_features)   # -> [B, 576, 4096]

        # B. 假设 text_embeds 输入是 [B, 10, 4096] 
        txt_proj = self.proj_text(text_embeds)

        # C. 在 dim = 1 维度拼接  (横向)
        # 结果为 [B, 10 + 576, 4096]
        fused_embeds = torch.cat([txt_proj, img_proj], dim = 1)

        return fused_embeds
    


# 实例化模型
model = ToyAlignmentModel()
model.eval()    # 不知道作用如何， 可以避免处理 Dropout/BatchNorm 节点

# 创建输入张量
dummy_text = torch.randn(1, 10, 4096)
dummy_image = torch.randn(1, 576, 1024)


# 导出 ONNX
output_file = "toy_vla_structure.onnx"
print(f"正在导出到{output_file} ...")


torch.onnx.export(
    model,                          # 要导出的模型
    (dummy_text, dummy_image),      # 模型的输入 (元组形式)
    output_file,                    # 输出文件名
    export_params=True,             # 是否带权重
    opset_version= 13,              # ONNX 版本
    do_constant_folding=True,       # 优化掉常量折叠
    input_names=['Text_Input', 'Image_Input'],              # 输入节点名称
    output_names=['Fused_Output'],                  # 输出节点
    dynamic_axes={                  # 关键！告诉ONNX哪些维度是可变的
        'Text_Input': {0: 'batch_size', 1:'seq_len'},
        'Image_Input': {0: 'batch_size'},
        'Fused_Output': {0: 'batch_size', 1: 'total_len'}
    }   
)


print("导出成功！")