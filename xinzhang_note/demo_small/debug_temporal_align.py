import torch
import torch.nn as nn
import math

class ToyTemporalAlign(nn.Module):
    def __init__(self, d_model = 16):
        super().__init__()
        self.d_model = d_model
        # 模拟 Ego-Pose 调制网络(FiLM 机制)
        # 从 Ego-Pose （4维) -> 生成 Gamma (16维) 和 Beta (16维)
        self.ego_proj = nn.Linear(4, d_model * 2)

    def nerg_pe(self, x):
        # 模拟 NeRF PE: 把 1 维坐标映射成多维 Sin / Cos
        # 此处简化逻辑，只做简单频率映射，在Netron中预期变为一排 Sin 和 Cos 节点
        freqs = [1.0, 2.0, 4.0, 8.0]
        features = []
        for freq in freqs:
            print("x * freq shape : ", (x*freq).shape)
            features.append(torch.sin(x * freq))
            features.append(torch.cos(x * freq))
        
        print("features.size()", len(features))

        # 拼接起来
        return torch.cat(features, dim = -1)
    

    def forward(self, current_query, memory_pe, ego_pose):
        # current_query: [B, 10, 16]
        # memory_pe : [B, 50, 8] <--- 假设原始 PE 是 8维， 4个频率对
        # ego_pose: [B, 50, 4]

        print("memory_pe shape : ", memory_pe.shape)

        #1. [重点观察] NeRF PE 结构
        # 主路： 空间信息(Spatial)
        # Input: 记忆中的特征，代表了“物体在哪里”，  
        # Status: 此时是在 t - 1 时刻的坐标系下描述的，如果不处理直接使用，会出现“重影" Ghosting
        pe_expanded = self.nerg_pe(memory_pe)   # [B, 50, 8] -> [B, 50, 64]   扩维

        print("pe_expanded shape", pe_expanded.shape)

        # 截断部分维度，方便演示结构
        pe_proj = pe_expanded[:, :, :self.d_model]


        # FiLM 调制结构， 双分支汇合结构
        
        
        # 分支 A: 控制信号 (Ego Pose)
        """
        一个简单的 MLP(ego_proj) 充当翻译官， 将输入(ego_pose 物理世界的 delta_x, delta_y, delta_theta)
        映射到 Gamma (16维) 和 Beta (16维) 作为 FiLM 这一仿射变换的参数。 

        物理直觉: 网络在此处决策，如果 ego_pose 是 (0, 0, 0)， 那么 Gamma 就是 (1, 1, 1, ...)， Beta 就是 (0, 0, 0, ...)，
        这意味着记忆中的特征不会被改变。 而如果 ego_pose 是 (1, 0, 0)， 那么 Gamma 就是 (1, 1, 1, ...)， Beta 就是 (1, 0, 0, ...)，
        这意味着记忆中的特征会被平移 1 单位。
        
        """
        
        style = self.ego_proj(ego_pose)   # [B, 50, 32]
        gamma, beta = style.chunk(2, dim=-1)        # 切分成 [B, 50, 16] 和 [B, 50, 16]


        # 分支 B: 原始信号 (Memory PE)
        # 融合操作: (1 + gamma) * x + beta
        # Netron 中显示 乘法(Mul) 和 加法(Add) 将两个分支连在一起 
        modulated_pe = pe_proj * (1 + gamma) + beta


        # 3. 最后拼接到 Current Query 上
        # 先把 modulated_pe 切割一部分 (Propagation)
        propagated_pe = modulated_pe[:, :5, :]   # 取前5个

        # 最终拼接
        output = torch.cat([current_query, propagated_pe], dim=1)

        return output
    
# 运行导出
model = ToyTemporalAlign(d_model=16)
model.eval()


# 构造输入
dummy_query = torch.randn(1, 10, 16)
dummy_mem_pe = torch.randn(1, 50, 1)        # 输入坐标 1 维
dummy_ego = torch.randn(1, 50, 4)           # Ego pose

torch.onnx.export(
    model, 
    (dummy_query, dummy_mem_pe, dummy_ego),
    "toy_temporal_align.onnx",
    input_names=['Query', 'Memory_Coord', 'Ego_Pose'],
    output_names=['Fused_Output'],
    opset_version=13
)

print("导出完成: toy_temporal_align.onnx")