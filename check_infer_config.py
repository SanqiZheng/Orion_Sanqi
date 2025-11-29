#!/usr/bin/env python3
"""
检查推理配置的完整性
确保所有必需字段都已配置
"""

def check_config(config_path):
    """检查配置文件的关键字段"""
    
    print(f"\n{'='*60}")
    print(f"检查配置: {config_path}")
    print('='*60)
    
    # 导入配置
    import sys
    import os
    sys.path.insert(0, os.getcwd())
    
    from mmcv import Config
    cfg = Config.fromfile(config_path)
    
    # 1. 检查data.test配置
    print("\n[1] data.test 配置:")
    if not hasattr(cfg.data, 'test'):
        print("  ❌ 缺少 data.test 配置")
        return False
    else:
        print(f"  ✅ type: {cfg.data.test.type}")
        print(f"  ✅ ann_file: {cfg.data.test.ann_file}")
        if hasattr(cfg.data.test, 'limit_samples'):
            print(f"  ✅ limit_samples: {cfg.data.test.limit_samples}")
        if hasattr(cfg.data.test, 'seq_mode'):
            print(f"  ✅ seq_mode: {cfg.data.test.seq_mode}")
    
    # 2. 检查模型配置
    print("\n[2] 模型配置:")
    print(f"  - tokenizer: {cfg.model.get('tokenizer', 'None')}")
    print(f"  - lm_head: {cfg.model.get('lm_head', 'None')}")
    print(f"  - use_gen_token: {cfg.model.get('use_gen_token', False)}")
    
    # 3. 检查bbox_coder
    print("\n[3] bbox_coder:")
    if 'pts_bbox_head' in cfg.model and 'bbox_coder' in cfg.model.pts_bbox_head:
        print(f"  ✅ type: {cfg.model.pts_bbox_head.bbox_coder.type}")
    else:
        print("  ⚠️  未配置 bbox_coder")
    
    # 4. 检查数据pipeline中的collect keys
    print("\n[4] 数据Pipeline Collect Keys:")
    pipeline = cfg.data.test.pipeline
    
    for step in pipeline:
        if step['type'] == 'LoadAnnotations3D':
            print(f"  - LoadAnnotations3D:")
            print(f"    with_bbox_3d: {step.get('with_bbox_3d', False)}")
            print(f"    with_label_3d: {step.get('with_label_3d', False)}")
            print(f"    with_attr_label: {step.get('with_attr_label', False)}")
            print(f"    with_light_state: {step.get('with_light_state', False)}")
        
        if step['type'] == 'MultiScaleFlipAug3D':
            for transform in step['transforms']:
                if transform['type'] == 'CustomCollect3D':
                    keys = transform['keys']
                    print(f"  - CustomCollect3D keys ({len(keys)} 个):")
                    
                    # 必需的keys
                    required_keys = [
                        'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                        'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
                        'ego_fut_cmd', 'ego_lcf_feat', 'can_bus', 'fut_valid_flag',
                        'lidar2img', 'cam_intrinsic', 'timestamp', 
                        'ego_pose', 'ego_pose_inv', 'command'
                    ]
                    
                    missing_keys = []
                    for key in required_keys:
                        if key in keys:
                            print(f"    ✅ {key}")
                        else:
                            print(f"    ❌ 缺少: {key}")
                            missing_keys.append(key)
                    
                    # 可选的keys（LLM相关）
                    optional_keys = ['input_ids', 'vlm_labels', 'traffic_state_mask', 'traffic_state']
                    print(f"  - 可选keys:")
                    for key in optional_keys:
                        if key in keys:
                            print(f"    ✅ {key}")
                    
                    if missing_keys:
                        print(f"\n  ⚠️  缺少 {len(missing_keys)} 个必需字段！")
                        return False
    
    # 5. 检查workflow
    print("\n[5] Workflow:")
    if hasattr(cfg, 'workflow'):
        print(f"  ✅ workflow: {cfg.workflow}")
        if cfg.workflow != [('test', 1)]:
            print(f"  ⚠️  警告: workflow应该是[('test', 1)]用于推理")
    else:
        print(f"  ⚠️  未配置workflow")
    
    print("\n" + "="*60)
    print("✅ 配置检查完成！")
    print("="*60)
    return True

if __name__ == '__main__':
    configs = [
        'adzoo/orion/configs/orion_stage3_infer_light.py',
        'adzoo/orion/configs/orion_stage3_infer_llm_vqa_light.py',
    ]
    
    all_ok = True
    for config in configs:
        try:
            if not check_config(config):
                all_ok = False
        except Exception as e:
            print(f"\n❌ 检查失败: {config}")
            print(f"   错误: {e}")
            all_ok = False
    
    if all_ok:
        print("\n" + "="*60)
        print("🎉 所有配置检查通过！可以运行推理")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ 部分配置有问题，请修复后再运行")
        print("="*60)

