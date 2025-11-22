import torch
# g1 = torch.Generator()           # 随机数生成器
# g1.manual_seed(42)

# 种子， 随机数生成器的“初始状态”。 使用相同种子，每次运行程序时会生成相同的随机数序列

# g2 = torch.Generator()
# g2.manual_seed(100)
# print(torch.rand(3, generator=g1))
# print(torch.rand(3, generator=g2))



# tensor2 = torch.tensor([10, 5, 8, 3, 12])


# boolean_mask = (tensor2 > 6)  # 比较操作生成布尔张量
# print(f"原始张量: {tensor2}")
# print(f"布尔掩码 (tensor2 > 6): {boolean_mask}")  #

# print(tensor2.sum())
# print(boolean_mask.sum())



import torch
import torch.nn as nn


# ===== 一些简单的 PE 函数 =====

def nerf_positional_encoding(x, num_freqs=4):
    """
    x: [..., D]
    返回: [..., D * 2 * num_freqs]
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    device = x.device

    freqs = 2 ** torch.arange(num_freqs, device=device).float()  # [F]
    # -> [..., D, F]
    x_expanded = x.unsqueeze(-1) * freqs

    # sin / cos -> [..., D, 2F]
    pe = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
    # 展平成 [..., D * 2F]
    pe = pe.reshape(*orig_shape[:-1], D * 2 * num_freqs)
    return pe


def pos2posemb1d(t, num_freqs=4):
    """
    一维时间戳的 PE：t: [..., 1] -> [..., 1 * 2 * num_freqs]
    """
    return nerf_positional_encoding(t, num_freqs=num_freqs)


# ===== ego pose PE 的一个简单实现 =====

class EgoPosePE(nn.Module):
    def __init__(self, hidden_dim, pose_pe_dim):
        super().__init__()
        self.proj = nn.Linear(pose_pe_dim, hidden_dim)

    def forward(self, pos, pose_pe):
        """
        pos:     [B, M, C]
        pose_pe: [B, M, pose_pe_dim]
        返回:     [B, M, C]
        """
        return pos + self.proj(pose_pe)


# ===== 主类：包含 temporal_alignment + 打印 shape =====

class ToyTemporalAlign(nn.Module):
    def __init__(self, B=2, Nq=4, M=6, C=32,
                 n_control=3, num_propagated=2,
                 num_freqs_pos=4, num_freqs_pose=4, num_freqs_time=4):
        super().__init__()

        self.B = B
        self.Nq = Nq
        self.M = M
        self.C = C
        self.n_control = n_control
        self.num_propagated = num_propagated
        self.with_ego_pos = True

        self.num_freqs_pos = num_freqs_pos
        self.num_freqs_pose = num_freqs_pose
        self.num_freqs_time = num_freqs_time

        # point cloud 范围: [x_min, y_min, z_min, x_max, y_max, z_max]
        self.pc_range = torch.tensor([-50.0, -50.0, -5.0,
                                       50.0,  50.0,  3.0])

        # ----- 历史 memory 玩具数据 -----
        # 历史 reference point: [B, M, 3]
        self.memory_reference_point = torch.rand(B, M, 3)

        # 历史特征: [B, M, C]
        self.memory_embedding = torch.randn(B, M, C)

        # 历史时间戳: [B, M, 1]，这里简单放个 0~1 等差
        ts = torch.linspace(0, 1, steps=M).view(1, M, 1)  # [1, M, 1]
        self.memory_timestamp = ts.repeat(B, 1, 1)        # [B, M, 1]

        # 历史 ego pose: [B, M, 4, 4]，这里用单位矩阵
        eye = torch.eye(4).view(1, 1, 4, 4)
        self.memory_egopose = eye.repeat(B, M, 1, 1)

        # ----- 各种 embedding 层 -----
        # 对 (归一化 ref_point * n_control) 的 PE 映射到 C
        ref_dim = 3 * n_control
        ref_pe_dim = ref_dim * 2 * num_freqs_pos
        self.query_pos = nn.Linear(ref_pe_dim, C)

        # ego pose PE: (time + pose_flat) 共有 13 维
        pose_in_dim = 1 + 12  # 1 time + 12 pose
        pose_pe_dim = pose_in_dim * 2 * num_freqs_pose
        self.ego_pose_pe = EgoPosePE(C, pose_pe_dim)

        # 时间 PE: 1 维时间 -> 2 * num_freqs_time
        time_in_dim = 1
        time_pe_dim = time_in_dim * 2 * num_freqs_time
        self.time_embedding = nn.Linear(time_pe_dim, C)

    def temporal_alignment(self, query_pos, tgt, reference_points):
        print(">>> 输入:")
        print("tgt:", tgt.shape)
        print("query_pos:", query_pos.shape)
        print("reference_points:", reference_points.shape)
        print("-" * 60)

        B = query_pos.size(0)

        # 1. 归一化历史 reference point
        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / \
                               (self.pc_range[3:6] - self.pc_range[0:3])
        print("[1] temp_reference_point:", temp_reference_point.shape)

        # 2. 对历史 ref_point 做 NeRF PE，并映射到 C 维
        rpt = temp_reference_point.repeat(1, 1, self.n_control)
        print("[2] temp_reference_point.repeat:", rpt.shape)

        temp_pos_pe = nerf_positional_encoding(rpt, num_freqs=self.num_freqs_pos)
        print("[3] after nerf_positional_encoding (temp_pos_pe):", temp_pos_pe.shape)

        temp_pos = self.query_pos(temp_pos_pe)
        print("[4] temp_pos after Linear:", temp_pos.shape)

        # 3. copy 历史 memory
        temp_memory = self.memory_embedding
        print("[5] temp_memory (memory_embedding):", temp_memory.shape)

        # 4. 初始 rec_ego_pose：单位矩阵
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0)
        rec_ego_pose = rec_ego_pose.repeat(B, query_pos.size(1), 1, 1)
        print("[6] rec_ego_pose init:", rec_ego_pose.shape)

        # 5. ego pose 分支
        if self.with_ego_pos:
            # 当前帧 ego_motion: time=0 + pose_flat
            zeros_t = torch.zeros_like(reference_points[..., :1])
            pose_flat = rec_ego_pose[..., :3, :].flatten(-2)
            rec_ego_motion = torch.cat([zeros_t, pose_flat], dim=-1)
            print("[7] rec_ego_motion (time+pose_flat):", rec_ego_motion.shape)

            rec_ego_motion_pe = nerf_positional_encoding(
                rec_ego_motion,
                num_freqs=self.num_freqs_pose
            )
            print("[8] rec_ego_motion_pe:", rec_ego_motion_pe.shape)

            # 历史 ego_motion: timestamp + pose_flat
            mem_pose_flat = self.memory_egopose[..., :3, :].flatten(-2)
            memory_ego_motion = torch.cat([self.memory_timestamp, mem_pose_flat], dim=-1).float()
            print("[9] memory_ego_motion (time+pose_flat):", memory_ego_motion.shape)

            memory_ego_motion_pe = nerf_positional_encoding(
                memory_ego_motion,
                num_freqs=self.num_freqs_pose
            )
            print("[10] memory_ego_motion_pe:", memory_ego_motion_pe.shape)

            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion_pe)
            print("[11] temp_pos after ego_pose_pe:", temp_pos.shape)

        # 6. 加时间编码：当前帧时间=0
        t0 = torch.zeros_like(reference_points[..., :1])
        t0_pe = pos2posemb1d(t0, num_freqs=self.num_freqs_time)
        print("[12] t0_pe:", t0_pe.shape)

        t0_emb = self.time_embedding(t0_pe)
        print("[13] t0_emb:", t0_emb.shape)

        query_pos = query_pos + t0_emb
        print("[14] query_pos after time_embedding:", query_pos.shape)

        # 历史的时间编码
        mem_t_pe = pos2posemb1d(self.memory_timestamp.float(), num_freqs=self.num_freqs_time)
        print("[15] mem_t_pe:", mem_t_pe.shape)

        mem_t_emb = self.time_embedding(mem_t_pe)
        print("[16] mem_t_emb:", mem_t_emb.shape)

        temp_pos = temp_pos + mem_t_emb
        print("[17] temp_pos after adding mem_t_emb:", temp_pos.shape)

        # 7. propagation: 拼接部分历史到当前
        if self.num_propagated > 0:
            K = self.num_propagated

            # tgt 拼接
            tgt = torch.cat([tgt, temp_memory[:, :K]], dim=1)
            print("[18] tgt after concat:", tgt.shape)

            # query_pos 拼接
            query_pos = torch.cat([query_pos, temp_pos[:, :K]], dim=1)
            print("[19] query_pos after concat:", query_pos.shape)

            # reference_points 拼接
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :K]],
                dim=1
            )
            print("[20] reference_points after concat:", reference_points.shape)

            # 这里源码是 query_pos.shape[1] + self.num_propagated
            # 按源码来写，方便你对比
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0)
            rec_ego_pose = rec_ego_pose.repeat(B, query_pos.shape[1] + K, 1, 1)
            print("[21] rec_ego_pose after re-init (注意这里的 +K):", rec_ego_pose.shape)

            # 更新剩余的历史 memory / pos
            temp_memory = temp_memory[:, K:]
            temp_pos = temp_pos[:, K:]
            print("[22] temp_memory remaining:", temp_memory.shape)
            print("[23] temp_pos remaining:", temp_pos.shape)

        print("=" * 60)
        print(">>> 返回:")
        print("tgt:", tgt.shape)
        print("query_pos:", query_pos.shape)
        print("reference_points:", reference_points.shape)
        print("temp_memory:", temp_memory.shape)
        print("temp_pos:", temp_pos.shape)
        print("rec_ego_pose:", rec_ego_pose.shape)
        print("=" * 60)

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose


if __name__ == "__main__":
    # 玩具超参数
    B, Nq, M, C = 2, 4, 6, 32
    n_control = 3
    num_propagated = 2

    model = ToyTemporalAlign(
        B=B,
        Nq=Nq,
        M=M,
        C=C,
        n_control=n_control,
        num_propagated=num_propagated,
    )

    # 当前帧的输入
    query_pos = torch.randn(B, Nq, C)
    tgt = torch.randn(B, Nq, C)
    reference_points = torch.rand(B, Nq, 3)

    # 跑一遍，看看全流程 shape
    outputs = model.temporal_alignment(query_pos, tgt, reference_points)
