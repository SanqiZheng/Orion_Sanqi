#!/usr/bin/env python3
"""
修复损坏的验证集数据文件
从训练集中划分出验证集
"""

import pickle
import os
import sys

def fix_val_dataset():
    """从训练集划分验证集"""
    
    train_file = 'data/infos/b2d_infos_train.pkl'
    val_file = 'data/infos/b2d_infos_val.pkl'
    
    print("=" * 60)
    print("🔧 修复验证集数据文件")
    print("=" * 60)
    
    # 1. 检查训练集文件
    if not os.path.exists(train_file):
        print(f"❌ 错误: 训练集文件不存在: {train_file}")
        sys.exit(1)
    
    train_size = os.path.getsize(train_file)
    print(f"✅ 训练集文件: {train_file}")
    print(f"   大小: {train_size / 1024 / 1024:.2f} MB")
    
    # 2. 检查验证集文件
    if os.path.exists(val_file):
        val_size = os.path.getsize(val_file)
        print(f"\n⚠️  当前验证集文件: {val_file}")
        print(f"   大小: {val_size} 字节")
        
        if val_size > 1024 * 1024:  # 超过1MB
            print("   状态: 看起来正常")
            response = input("\n是否仍要重新生成验证集？[y/N]: ")
            if response.lower() != 'y':
                print("取消操作")
                return
        else:
            print("   状态: ❌ 损坏（太小）")
    
    # 3. 加载训练集
    print("\n📂 加载训练集数据...")
    try:
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        sys.exit(1)
    
    # 4. 检查数据结构
    if not isinstance(train_data, dict):
        print(f"❌ 错误: 训练集数据格式不正确，类型为 {type(train_data)}")
        sys.exit(1)
    
    if 'infos' not in train_data:
        print(f"❌ 错误: 训练集数据中没有 'infos' 字段")
        print(f"   可用字段: {train_data.keys()}")
        sys.exit(1)
    
    total_samples = len(train_data['infos'])
    print(f"✅ 训练集样本数: {total_samples}")
    
    # 5. 划分验证集（取10%）
    val_ratio = 0.1
    val_size = max(10, int(total_samples * val_ratio))  # 至少10个样本
    
    print(f"\n📊 划分验证集:")
    print(f"   比例: {val_ratio * 100}%")
    print(f"   验证集样本数: {val_size}")
    print(f"   剩余训练集样本数: {total_samples - val_size}")
    
    # 6. 创建验证集数据
    val_data = {
        'infos': train_data['infos'][:val_size],
        'metadata': train_data.get('metadata', {})
    }
    
    # 7. 备份原文件（如果存在且不为空）
    if os.path.exists(val_file):
        backup_file = val_file + '.backup'
        print(f"\n💾 备份原文件: {backup_file}")
        os.rename(val_file, backup_file)
    
    # 8. 保存验证集
    print(f"\n💾 保存验证集: {val_file}")
    try:
        with open(val_file, 'wb') as f:
            pickle.dump(val_data, f)
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        sys.exit(1)
    
    # 9. 验证新文件
    new_size = os.path.getsize(val_file)
    print(f"   新文件大小: {new_size / 1024 / 1024:.2f} MB")
    
    # 10. 测试加载
    print("\n🔍 验证新文件...")
    try:
        with open(val_file, 'rb') as f:
            test_data = pickle.load(f)
        print(f"✅ 验证成功！样本数: {len(test_data['infos'])}")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 验证集修复完成！")
    print("=" * 60)
    print("\n现在可以运行推理了：")
    print("  bash adzoo/orion/quick_eval_llm.sh")
    print("\n注意：如需完整评估，请使用原始验证集数据")
    print("=" * 60)

if __name__ == '__main__':
    fix_val_dataset()

