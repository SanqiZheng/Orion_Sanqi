"""
ORION Stage1 训练配置 - 完整LLM版本
包含VQA功能，需要大内存（推荐32GB+）

⚠️  警告：此配置会加载13GB的LLM模型
   - 需要至少32GB系统内存
   - 如果内存不足会导致系统死机
   - 流程测试请使用 orion_stage1_train_minimal.py
"""

_base_ = ["../_base_/datasets/nus-3d.py",
          "../_base_/default_runtime.py"]

backbone_norm_cfg = dict(type='LN', requires_grad=True)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

map_classes = ['Broken','Solid','SolidSolid','Center','TrafficLight','StopSign']
map_fixed_ptsnum_per_gt_line = 11
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
past_frames = 2
future_frames = 6
_dim_ = 256
_ffn_dim_ = _dim_*2

ida_aug_conf = {
        "resize_lim": (0.37, 0.45),
        "final_dim": (320, 640),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# 保持与官方一致的 NameMapping（简化版）
NameMapping = {
    # 只保留核心类别的映射
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.audi.etron": 'car',
    'traffic.speed_limit.30': 'traffic_sign',
    'traffic.traffic_light': 'traffic_light',
    'static.prop.warningconstruction': 'traffic_cone',
    'walker.pedestrian.0001': 'pedestrian',
    'static.prop.dirtdebris01': 'others',
}

# 评估配置
eval_cfg = dict(
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    min_recall=0.1,
    min_precision=0.1,
    mean_ap_weight=5,
    class_names=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian'],
    tp_metrics=['trans_err', 'scale_err', 'orient_err', 'vel_err'],
    err_name_maping=dict(
        trans_err='mATE',
        scale_err='mASE',
        orient_err='mAOE',
        vel_err='mAVE',
        attr_err='mAAE'
    ),
    class_range=dict(
        car=(50, 50),
        van=(50, 50),
        truck=(50, 50),
        bicycle=(40, 40),
        traffic_sign=(30, 30),
        traffic_cone=(30, 30),
        traffic_light=(30, 30),
        pedestrian=(40, 40)
    )
)

queue_length = 1
predict_steps = 12
predict_modes = 6
use_nonlinear_optimizer = True
use_memory = True

# 🔥 关键：最小化配置
num_gpus = 1
batch_size = 1
num_iters_per_epoch = 3  # 只运行3个iter测试流程
num_epochs = 1  # 只训练1个epoch

llm_path = 'ckpts/pretrain_qformer/'
use_gen_token = False
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']
pretrain = True
use_col_loss = False

# 数据配置
dataset_type = 'B2DOrionDataset'
data_root = 'data/bench2drive'
info_root = 'data/infos'
map_root = 'data/bench2drive/maps'
map_file = 'data/infos/b2d_map_infos.pkl'
ann_file_train = 'data/infos/b2d_infos_train.pkl'
ann_file_val = 'data/infos/b2d_infos_val.pkl'

# Pipeline 配置
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True,
        with_light_state=True),
    dict(
        type='VADObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='VADObjectNameFilter',
        classes=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
                'traffic_light', 'pedestrian', 'others']),
    # ✅ 启用VQA数据加载
    dict(
        type='LoadAnnoatationVQA',
        base_desc_path='./data/chat-B2D/train',
        tokenizer=llm_path,
        max_length=2048,
        use_gen_token=use_gen_token,
        pretrain=pretrain),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=ida_aug_conf,
        training=True),
    dict(
        type='ResizeMultiview3D',
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode='value'),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='PETRFormatBundle3D',
        class_names=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
                    'traffic_light', 'pedestrian', 'others'],
        collect_keys=collect_keys),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'input_ids', 'vlm_labels',  # VQA相关，已启用
            'gt_attr_labels', 'ego_fut_trajs', 'ego_fut_masks',
            'ego_fut_cmd', 'ego_lcf_feat', 'can_bus',
            'traffic_state_mask', 'traffic_state', 'lidar2img',
            'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command'
        ])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='VADObjectRangeFilter',
        point_cloud_range=point_cloud_range),
    dict(
        type='VADObjectNameFilter',
        classes=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
                'traffic_light', 'pedestrian', 'others']),
    dict(
        type='ResizeCropFlipRotImage',
        data_aug_conf=ida_aug_conf,
        training=False),
    dict(
        type='ResizeMultiview3D',
        img_scale=(640, 640),
        keep_ratio=False,
        multiscale_mode='value'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='PadMultiViewImage', size_divisor=32),
    # ✅ 启用VQA测试加载
    dict(
        type='LoadAnnoatationCriticalVQATest',
        load_type=['critical_qa'],
        tokenizer=llm_path,
        use_gen_token=use_gen_token,
        max_length=2048),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=['car', 'van', 'truck', 'bicycle', 'traffic_sign',
                            'traffic_cone', 'traffic_light', 'pedestrian', 'others'],
                with_label=False),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'input_ids', 'vlm_labels',  # VQA相关，已启用
                    'gt_attr_labels', 'ego_fut_trajs',
                    'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                    'can_bus', 'fut_valid_flag', 'lidar2img',
                    'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command'
                ])
        ])
]

# 🔥 数据集配置 - 极少量样本
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # 单进程加载
    train=dict(
        type='B2DOrionDataset',
        data_root=data_root,
        ann_file=ann_file_train,
        limit_samples=5,  # ⚠️ 只加载5个训练样本
        pipeline=train_pipeline,
        classes=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        seq_mode=True,
        seq_split_num=1,
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        queue_length=queue_length,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
    ),
    val=dict(
        type='B2DOrionDataset',
        data_root=data_root,
        ann_file=ann_file_val,
        limit_samples=3,  # ⚠️ 只加载3个验证样本
        pipeline=test_pipeline,
        classes=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        name_mapping=NameMapping,
        map_root=map_root,
        map_file=map_file,
        queue_length=queue_length,
        past_frames=past_frames,
        future_frames=future_frames,
        point_cloud_range=point_cloud_range,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        eval_cfg=eval_cfg,
    ),
)

# 模型配置
model = dict(
    type='Orion',
    save_path='./results_planning_only/',
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    # ✅ 启用完整LLM功能
    # 需要加载13GB的LLM模型权重
    tokenizer=llm_path,  # 启用tokenizer
    lm_head=llm_path,    # 启用LLM
    use_gen_token=use_gen_token,  # 使用配置值
    use_diff_decoder=False,
    use_col_loss=use_col_loss,
    
    loss_plan_reg=dict(type='L1Loss', loss_weight=3.0),
    loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=3.0, dis_thresh=1.0),
    loss_vae_gen=dict(type='ProbabilisticLoss', loss_weight=3.0),
    
    img_backbone=dict(
        type='EVAViT',
        img_size=640,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=2.6666666666666665,
        window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22],
        qkv_bias=True,
        drop_path_rate=0.3,
        flash_attn=True,
        with_cp=True,  # 启用梯度检查点，节省显存
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
            flash_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='LaneHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.02),
                iou_cost=dict(type='IoUCost', weight=0.0))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.02),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.0)),
    
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
        use_memory=True,
        use_col_loss=use_col_loss,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        memory_decoder_transformer=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=512,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False),
        motion_transformer_decoder=dict(
            type='OrionTransformerDecoder',
            num_layers=1,
            embed_dims=256,
            num_heads=8,
            dropout=0.0,
            feedforward_dims=512,
            with_cp=True,
            flash_attn=True,
            return_intermediate=False),
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        score_threshold=0.2,
        class_agnostic_nms=dict(
            classes=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            compensate=[0, 0, 0.3, 0, 0, 0, 0, 0.3, 0],
            pre_max_size=1000,
            post_max_size=300,
            nms_thr=0.1),
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
            flash_attn=True),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_traffic=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    
    # 训练配置
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))),
    
    freeze_backbone=True  # 冻结backbone
)

# 训练配置
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=8e-5,
    betas=(0.9, 0.999),
    weight_decay=1e-5,
    paramwise_cfg=dict(
        decay_rate=0.9,
        head_decay_rate=4.0,
        lm_head_decay_rate=0.1,
        decay_type='vit_wise',
        num_layers=24
    )
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2)
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,  # 减少warmup迭代
    warmup_ratio=1./3.,
    min_lr_ratio=1e-3
)

# Runner配置
total_iters = num_iters_per_epoch * num_epochs
runner = dict(type='IterBasedRunner', max_iters=total_iters)

# 检查点和日志配置
checkpoint_config = dict(interval=total_iters, max_keep_ckpts=1)  # 只在最后保存
log_config = dict(
    interval=1,  # 每个iter都记录
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# 评估配置
evaluation = dict(interval=total_iters, pipeline=None)  # 只在最后评估

# 其他配置
find_unused_parameters = False
fp16 = dict(loss_scale='dynamic')
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'ckpts/eva02_petr_proj.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = range(0, num_gpus)

print("=" * 80)
print("⚠️  完整训练配置已加载（包含LLM和VQA）")
print(f"训练样本数: 5")
print(f"验证样本数: 3")
print(f"迭代次数: {total_iters}")
print(f"Batch Size: {batch_size}")
print("🔥 LLM加载: 已启用（需要~13GB内存）")
print("   需要至少32GB系统内存，否则可能死机！")
print("   如果内存不足，请使用 orion_stage1_train_minimal.py")
print("=" * 80)

