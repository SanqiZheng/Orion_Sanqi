"""
unsloth 对LLM的算子和反向传播(计算梯度？)进行优化

"""

import imp
from unsloth import FastLanguageModel
import torch, os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

max_seq_length = 2048   # 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,                   # Unsloth默认使用 torch.bfloat16, 可以设置为 torch.float16 或 torch.float32
    load_in_4bit = True,            # 4bit 量化， 节省显存
)


FastLanguageModel.for_inference(model)          # 切换到推理模式， 开启优化
model.to("cuda")        # 切换到cuda， 开启GPU加速



# 简单对话
def chat(prompt: str, max_new_tokens: int = 128):
    # 对输入文本进行编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,              # 最大生成 token 数
            do_sample=True,                             # 开启采样， 否则会生成重复的 token
            top_p=0.95,                                 #  nucleus sampling， 只考虑前 95% 概率最高的 token
            top_k=50,                                   # 只考虑前 50 个概率最高的 token
            temperature=0.7,                            # 温度， 控制输出的随机性， 0.7 是一个常用值
            eos_token_id=tokenizer.eos_token_id,        # 结束符
            )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response