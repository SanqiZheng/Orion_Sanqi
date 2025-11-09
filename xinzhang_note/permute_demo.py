"""
permute和hidden_states的演示代码
这个文件可以帮助你理解permute函数和hidden_states的转换过程
"""

import torch
import torch.nn as nn

def demo_permute_basic():
    """演示permute的基本用法"""
    print("=" * 60)
    print("1. permute基本用法演示")
    print("=" * 60)
    
    # 创建一个简单的3D张量
    x = torch.randn(2, 3, 4)
    print(f"原始张量形状: {x.shape}")
    print(f"维度含义: (batch=2, height=3, width=4)")
    
    # 使用permute重新排列
    y = x.permute(2, 0, 1)  # 将 (2,3,4) 变成 (4,2,3)
    print(f"\npermute(2, 0, 1) 后: {y.shape}")
    print(f"维度含义: (width=4, batch=2, height=3)")
    print(f"数据是否相同: {torch.equal(x, y)}")  # False，因为形状不同
    print(f"但元素总数相同: {x.numel() == y.numel()}")  # True


def demo_hidden_states_transformation():
    """演示hidden_states在ORION中的转换过程"""
    print("\n" + "=" * 60)
    print("2. hidden_states转换过程演示（模拟ORION代码）")
    print("=" * 60)
    
    # 模拟batch_size=2的情况
    batch_size = 2
    feature_dim = 4096
    layer_dim = 4
    
    # 步骤1: 从LLM获取ego_feature
    print("\n步骤1: 从LLM提取ego_feature")
    ego_feature = torch.randn(batch_size, feature_dim)
    print(f"ego_feature形状: {ego_feature.shape}")
    print(f"含义: (batch_size={batch_size}, feature_dim={feature_dim})")
    
    # 步骤2: 添加序列维度
    print("\n步骤2: unsqueeze(1) - 添加序列长度维度")
    hidden_states = ego_feature.unsqueeze(1)
    print(f"hidden_states形状: {hidden_states.shape}")
    print(f"含义: (batch_size={batch_size}, seq_len=1, feature_dim={feature_dim})")
    print(f"为什么需要: GRU等序列模型需要序列长度维度")
    
    # 步骤3: permute重新排列维度
    print("\n步骤3: permute(1, 0, 2) - 重新排列维度顺序")
    print(f"转换前: {hidden_states.shape}")
    print(f"维度顺序: (batch, seq_len, feature)")
    
    hidden_states = hidden_states.permute(1, 0, 2)
    print(f"转换后: {hidden_states.shape}")
    print(f"维度顺序: (seq_len, batch, feature)")
    print(f"为什么需要: PyTorch的GRU期望输入格式为 (seq_len, batch, features)")
    
    # 步骤4: reshape为多层GRU准备
    print("\n步骤4: reshape - 为多层GRU准备初始隐藏状态")
    print(f"转换前: {hidden_states.shape}")
    hidden_state = hidden_states.reshape(layer_dim, -1, int(feature_dim/layer_dim))
    print(f"转换后: {hidden_state.shape}")
    print(f"含义: (num_layers={layer_dim}, batch_size={batch_size}, hidden_dim={int(feature_dim/layer_dim)})")
    print(f"为什么需要: 多层GRU需要为每一层提供独立的初始隐藏状态")
    
    # 验证数据没有丢失
    print(f"\n验证: 元素总数是否相同?")
    print(f"ego_feature元素数: {ego_feature.numel()}")
    print(f"hidden_state元素数: {hidden_state.numel()}")
    print(f"是否相等: {ego_feature.numel() == hidden_state.numel()}")


def demo_permute_vs_transpose():
    """演示permute和transpose的区别"""
    print("\n" + "=" * 60)
    print("3. permute vs transpose 对比")
    print("=" * 60)
    
    x = torch.randn(2, 3, 4)
    print(f"原始张量: {x.shape}")
    
    # transpose只能交换两个维度
    y1 = x.transpose(0, 1)  # 交换维度0和1
    print(f"\ntranspose(0, 1): {y1.shape}")
    print(f"含义: 交换第0维和第1维")
    
    # permute可以重新排列所有维度
    y2 = x.permute(2, 0, 1)  # 重新排列所有维度
    print(f"\npermute(2, 0, 1): {y2.shape}")
    print(f"含义: 将(0,1,2)维重新排列为(2,0,1)")
    
    print("\n总结: permute更灵活，可以同时重新排列多个维度")


def demo_visualize_transformation():
    """可视化维度转换过程"""
    print("\n" + "=" * 60)
    print("4. 维度转换可视化")
    print("=" * 60)
    
    batch_size = 4
    feature_dim = 4096
    layer_dim = 4
    
    # 创建有意义的张量（用索引标记，方便追踪）
    ego_feature = torch.arange(batch_size * feature_dim).reshape(batch_size, feature_dim)
    print(f"初始 ego_feature: {ego_feature.shape}")
    print(f"前几个元素: {ego_feature[0, :5]}")
    
    # 转换过程
    step1 = ego_feature.unsqueeze(1)
    print(f"\n步骤1 - unsqueeze(1): {step1.shape}")
    print(f"step1[0, 0, :5] = {step1[0, 0, :5]}")
    
    step2 = step1.permute(1, 0, 2)
    print(f"\n步骤2 - permute(1, 0, 2): {step2.shape}")
    print(f"step2[0, 0, :5] = {step2[0, 0, :5]}")  # 应该和step1[0, 0, :5]相同
    
    step3 = step2.reshape(layer_dim, -1, int(feature_dim/layer_dim))
    print(f"\n步骤3 - reshape: {step3.shape}")
    print(f"step3[0, 0, :5] = {step3[0, 0, :5]}")  # 前1024个元素
    
    # 验证数据完整性
    print(f"\n数据完整性验证:")
    print(f"ego_feature元素总数: {ego_feature.numel()}")
    print(f"step3元素总数: {step3.numel()}")
    print(f"数据是否完整: {ego_feature.numel() == step3.numel()}")


def demo_gru_input_format():
    """演示为什么需要permute来适配GRU"""
    print("\n" + "=" * 60)
    print("5. 为什么需要permute适配GRU？")
    print("=" * 60)
    
    # 创建一个简单的GRU
    gru = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=False)
    
    # GRU的默认输入格式 (seq_len, batch, features)
    seq_len, batch_size, input_size = 5, 3, 10
    x_default = torch.randn(seq_len, batch_size, input_size)
    print(f"GRU默认输入格式: {x_default.shape}")
    print(f"含义: (seq_len={seq_len}, batch={batch_size}, features={input_size})")
    
    # 如果我们有 (batch, seq_len, features) 格式的数据
    x_batch_first = torch.randn(batch_size, seq_len, input_size)
    print(f"\n常见的数据格式: {x_batch_first.shape}")
    print(f"含义: (batch={batch_size}, seq_len={seq_len}, features={input_size})")
    
    # 需要使用permute转换
    x_converted = x_batch_first.permute(1, 0, 2)
    print(f"\n使用permute转换后: {x_converted.shape}")
    print(f"现在可以输入GRU了！")
    
    # 或者使用batch_first=True
    gru_batch_first = nn.GRU(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
    print(f"\n或者使用batch_first=True的GRU，可以直接使用原始格式")


if __name__ == "__main__":
    print("permute和hidden_states演示程序")
    print("=" * 60)
    
    # 运行所有演示
    demo_permute_basic()
    demo_hidden_states_transformation()
    demo_permute_vs_transpose()
    demo_visualize_transformation()
    demo_gru_input_format()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n提示: 你可以修改这些代码，尝试不同的参数来加深理解。")

