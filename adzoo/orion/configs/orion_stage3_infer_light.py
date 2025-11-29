"""
ORION Stage3 推理配置 - 轻量版（不带LLM）
基于 orion_stage3_infer_llm_vqa_light.py，禁用LLM以节省资源
适用于只需要检测和规划，不需要VQA的场景（7-10GB显存 + 12-15GB内存）
"""

_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

backbone_norm_cfg = dict(type='LN', requires_grad=True)

# 点云范围
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 地图类别
map_classes = ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign']

map_fixed_ptsnum_per_gt_line = 11
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
past_frames = 2
future_frames = 6
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2

# 图像增强配置
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

# 占用流网格配置
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# 车辆名称映射（简化版）
NameMapping = {
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.tesla.model3": 'car',
    "vehicle.ford.mustang": 'car',
    # van
    "vehicle.ford.ambulance": "van",
    "vehicle.mercedes.sprinter": "van",
    "vehicle.volkswagen.t2": 'van',
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    "vehicle.ford.f150": 'truck',
    # traffic sign
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.stop": 'traffic_sign',
    # traffic light
    "traffic.traffic_light": 'traffic_light',
    # traffic cone
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.constructioncone": 'traffic_cone',
    # pedestrian
    "walker.pedestrian.0001": 'pedestrian',
    # others
    "static.prop.dirtdebris01": 'others',
}

# 类别名称
class_names = [
    'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
]

# 评估配置
eval_cfg = {
    "dist_ths": [0.5, 1.0, 2.0, 4.0],
    "dist_th_tp": 2.0,
    "min_recall": 0.1,
    "min_precision": 0.1,
    "mean_ap_weight": 5,
    "class_names":['car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian'],
    "tp_metrics":['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    "err_name_maping":{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'},
    "class_range":{'car':(50,50),'van':(50,50),'truck':(50,50),'bicycle':(40,40),'traffic_sign':(30,30),'traffic_cone':(30,30),'traffic_light':(30,30),'pedestrian':(40,40)}
}

# ============================================
# 🔥 关键配置：轻量推理（不带LLM）
# ============================================
use_memory = True
num_gpus = 1  # 单GPU推理
batch_size = 1  # 最小batch_size
use_gen_token = False  # ❌ 禁用生成token
use_col_loss = False   # ❌ 禁用碰撞损失
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

# 模型配置
model = dict(
    type='Orion',
    save_path='./results_infer_light/',  # 推理结果保存路径
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    # 🔥 禁用LLM（节省7-8GB显存和10-15GB内存）
    tokenizer=None,
    lm_head=None,
    use_gen_token=False,
    use_diff_decoder=False, 
    use_col_loss=False,
    
    # 图像骨干网络
    img_backbone=dict(
        type='EVAViT',
        img_size=640, 
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4*2/3,
        window_block_indexes=(
            list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + 
            list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + 
            list(range(18, 20)) + list(range(21, 23))
        ),
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True,  # 启用梯度检查点
        frozen=False,
    ), 
    
    # 地图检测头
    map_head=dict(
        type='OrionHeadM',
        num_classes=6,
        in_channels=1024,
        out_dims=4096,
        memory_len=600,
        with_mask=True,
        topk_proposals=300,
        num_lane=1800,
        num_lanes_one2one=300,
        k_one2many=5,
        lambda_one2many=1.0,
        num_extra=256,
        n_control=11,
        pc_range=point_cloud_range,
        code_weights=[1.0, 1.0],
        score_threshold=0.2,
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        )
    ),
    
    # 3D目标检测头
    pts_bbox_head=dict(
        type='OrionHead',
        num_classes=9,
        in_channels=1024,
        out_dims=4096,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=256,
        n_control=11,
        match_with_velo=False,
        pred_traffic_light_state=True,
        use_col_loss=False,
        use_memory=use_memory,
        scalar=10,
        noise_scale=1.0, 
        dn_weight=1.0,
        split=0.75,
        use_pe=False,
        
        # 运动解码器
        motion_transformer_decoder=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False,
        ),
        
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        score_threshold=0.2,
        class_agnostic_nms=dict(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8], 
            compensate=[0, 0, 0.3, 0, 0, 0, 0, 0.3, 0],
            pre_max_size=1000,
            post_max_size=300,
            nms_thr=0.1,
        ),
        
        # 内存解码器
        memory_decoder_transformer=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=_dim_,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=_ffn_dim_,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False
        ),
        
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
        
        bbox_coder=dict(
            type='NMSFreeCoder',  # 🔥 修复：使用与训练一致的coder
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=9
        )
    ),
)

# 数据集配置
dataset_type = "B2DOrionDataset"
data_root = "data/bench2drive"
info_root = "data/infos"
map_root = "data/bench2drive/maps"
map_file = "data/infos/b2d_map_infos.pkl"

file_client_args = dict(backend="disk")
# 🔥 临时修复：验证集文件损坏（只有5字节），使用训练集数据代替
ann_file_test = info_root + "/b2d_infos_train.pkl"  # 原值: b2d_infos_val.pkl

# 测试数据处理流程（不加载VQA数据）
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True, with_light_state=True),
    dict(type='VADObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='VADObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # 🔥 不加载VQA数据（节省内存和处理时间）
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False
            ),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'can_bus', 'fut_valid_flag'
                ] + collect_keys,
            )
        ]
    )
]

# 数据加载器配置
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,  # 减少worker数量
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        limit_samples=10,  # 🔥 添加：只推理10个样本（快速测试）
        pipeline=test_pipeline,
        classes=class_names,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        modality=input_modality,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        eval_cfg=eval_cfg,
        test_mode=True,
        box_type_3d='LiDAR',
        seq_mode=False,  # 🔥 添加：关闭序列模式
    ),
    nonshuffler_sampler=dict(type="DistributedSampler", shuffle=False),
)

# 日志配置
log_config = dict(
    interval=10, 
    hooks=[
        dict(type="TextLoggerHook"), 
        dict(type="TensorboardLoggerHook")
    ]
)

# 推理专用配置
find_unused_parameters = False
fp16 = dict(loss_scale='dynamic')  # 启用混合精度推理
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('test', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'

print("=" * 80)
print("⚠️  ORION推理配置已加载（轻量版，不带LLM）")
print(f"GPU数量: {num_gpus}")
print(f"Batch Size: {batch_size}")
print("🔥 优化配置:")
print("  - 单GPU推理")
print("  - LLM: 已禁用（节省资源）")
print("  - VQA: 已禁用")
print("  - 混合精度: 已启用")
print("  - 梯度检查点: 已启用")
print("⚠️  预计资源占用: 7-10GB显存 + 12-15GB内存")
print("💡 如需LLM和VQA功能，请使用: orion_stage3_infer_llm_vqa_light.py")
print("=" * 80)

