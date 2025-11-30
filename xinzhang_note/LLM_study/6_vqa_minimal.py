import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) 准备权重路径
VISION_BACKBONE = "/path/to/vision_encoder"     # 如 EVA/ViT/CLIP
LLM_BACKBONE    = "/path/to/llm"                # 如 Vicuna/LLaMA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) 图像预处理
img = Image.open("demo.jpg").convert("RGB")
img_tf = transforms.Compose([
    transforms.Resize((320, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
img_tensor = img_tf(img).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

# 3) 视觉编码（假设输出 [B, N, C]）
vision_model = ...  # load your ViT/CLIP vision tower
with torch.no_grad():
    vision_tokens = vision_model(img_tensor)      # [1, N, C]

# 4) 特征对齐/压缩到 LLM 维度（例：线性+平均或 Q-Former）
proj = torch.nn.Linear(vision_tokens.size(-1), 4096).to(DEVICE)
vision_embeds = proj(vision_tokens)               # [1, N, D_llm]

# 5) 文本编码
tokenizer = AutoTokenizer.from_pretrained(LLM_BACKBONE, use_fast=True)
question = "What is the color of the traffic light?"
prompt = f"USER: <img> {question}\nASSISTANT:"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# 6) 将视觉 token 塞进 LLM（两种常见做法示意）
# 6a 前缀方式：把视觉 embed 当作额外 prefix，需定制 LLM 的 forward 支持 inputs_embeds
llm = AutoModelForCausalLM.from_pretrained(LLM_BACKBONE).to(DEVICE)
text_embeds = llm.get_input_embeddings()(inputs.input_ids)
inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # [B, N+T, D]
attention_mask = torch.cat([
    torch.ones(vision_embeds.size()[:2], device=DEVICE), 
    inputs.attention_mask
], dim=1)

# 7) 解码生成
gen_ids = llm.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    max_new_tokens=64,
    do_sample=False
)

# 8) 后处理
# 去掉前缀视觉 token 对应的占位，保留文本段后的新 token
gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
# 可按分隔符截断：取 "ASSISTANT:" 后面的部分
answer = gen_text.split("ASSISTANT:")[-1].strip()
print("Answer:", answer)

"""
关键点说明

视觉特征压缩：若用 Q-Former，先把视觉 token 喂入 cross-attn，把 K 个 learnable query 取出，再投影到 LLM 维度；这里用线性+拼接示意。
视觉与文本对齐：LLM 需要同维度的输入，常见做法是 inputs_embeds（前缀/插槽），或改 LLM 的 cross-attn 层以视觉 token 做 kv。
Prompt 设计：<img> 只是示意，真实模型用自己约定的特殊 token；保持“USER/ASSISTANT”或你模型的对话模板。
后处理：去掉 BOS/EOS/特殊标记，按分隔符截答案；必要时 strip 空格、截到第一个换行。
资源控制：用 fp16、max_new_tokens 小、短 prompt，batch=1，避免爆显存。
一次完整 VQA 推理步骤（按执行顺序）

读图、预处理。
视觉编码（ViT/CLIP/EVA）→ patch token。
对齐/压缩视觉 token 到 LLM 维度与长度（线性或 Q-Former）。
文本分词，拼 prompt。
构造多模态输入：视觉 token 前缀/跨注意力 + 文本 token。
LLM 自回归生成答案 token。
解码为字符串，去除特殊符号，截断到 EOS/分隔符。
把上面示例替换成你本地的视觉塔权重和 LLM 权重，就能跑出“一图一问一答”的最小 VQA demo。

"""