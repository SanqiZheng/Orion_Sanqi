#!/usr/bin/env python3
"""
创建最小的虚拟数据文件，用于代码学习
避免因缺少数据文件导致程序提前退出
"""

import pickle
import os

print("🔧 创建虚拟数据文件...")

# 创建目录
os.makedirs('data/infos', exist_ok=True)

# 创建最小的训练数据标注（只包含1个样本）
train_infos = {
    'infos': [{
        'token': 'dummy_000001',
        'timestamp': 0,
        'ego2global_translation': [0, 0, 0],
        'ego2global_rotation': [1, 0, 0, 0],
        'cams': {
            f'CAM_{i}': {
                'data_path': f'dummy_cam_{i}.jpg',
                'sensor2ego_translation': [0, 0, 0],
                'sensor2ego_rotation': [1, 0, 0, 0],
                'cam_intrinsic': [[1000, 0, 800], [0, 1000, 450], [0, 0, 1]],
            }
            for i in range(6)
        },
        'gt_boxes': [],
        'gt_names': [],
        'gt_velocity': [],
        'num_lidar_pts': [],
        'num_radar_pts': [],
    }],
    'metadata': {
        'version': 'v1.0-dummy',
        'comment': 'Dummy data for code learning'
    }
}

# 保存文件
train_file = 'data/infos/b2d_infos_train.pkl'
val_file = 'data/infos/b2d_infos_val.pkl'

with open(train_file, 'wb') as f:
    pickle.dump(train_infos, f)
print(f"✓ 创建: {train_file}")

with open(val_file, 'wb') as f:
    pickle.dump(train_infos, f)
print(f"✓ 创建: {val_file}")

# 创建地图信息文件
map_infos = {
    'scene_1': {
        'lanes': [],
        'boundaries': []
    }
}

map_file = 'data/infos/b2d_map_infos.pkl'
with open(map_file, 'wb') as f:
    pickle.dump(map_infos, f)
print(f"✓ 创建: {map_file}")

print("\n✅ 虚拟数据文件创建完成！")
print("⚠️  注意：这些是虚拟数据，仅用于代码流程测试")

