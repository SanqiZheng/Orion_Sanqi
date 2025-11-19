"""
ORION Stage1 训练配置 - 最小化版本
用于流程测试和学习，大幅减少资源占用
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
NameMapping = dict({
    'vehicle.bh.crossbike': 'bicycle',
    'vehicle.diamondback.century': 'bicycle',
    'vehicle.gazelle.omafiets': 'bicycle',
    'vehicle.audi.etron': 'car',
    'vehicle.chevrolet.impala': 'car',
    'vehicle.dodge.charger_2020': 'car',
    'vehicle.dodge.charger_police': 'car',
    'vehicle.dodge.charger_police_2020': 'car',
    'vehicle.lincoln.mkz_2017': 'car',
    'vehicle.lincoln.mkz_2020': 'car',
    'vehicle.mini.cooper_s_2021': 'car',
    'vehicle.mercedes.coupe_2020': 'car',
    'vehicle.ford.mustang': 'car',
    'vehicle.nissan.patrol_2021': 'car',
    'vehicle.audi.tt': 'car',
    'vehicle.ford.crown': 'car',
    'vehicle.tesla.model3': 'car',
    'vehicle.tesla.model3_2020': 'car',
    'vehicle.harley-davidson.lowrider': 'motorcycle',
    'vehicle.ford.ambulance': 'van',
    'vehicle.carlamotors.firetruck': 'truck',
    'vehicle.ford.ambulance01': 'van',
    'vehicle.ford.ambulance02': 'van',
    'vehicle.mini.cooper_s': 'car',
    'vehicle.bmw.isetta': 'car',
    'vehicle.nissan.patrol_2020': 'car',
    'vehicle.jeep.wrangler_rubicon': 'car',
    'vehicle.mercedes.sprinter': 'van',
    'vehicle.toyota.prius': 'car',
    'vehicle.yamaha.yzf': 'motorcycle',
    'vehicle.dodge.pickup': 'car',
    'vehicle.ford.transit': 'van',
    'vehicle.mercedes.coupe': 'car',
    'vehicle.gmc.pickup': 'car',
    'vehicle.volkswagen.t2': 'van',
    'vehicle.bmw.130': 'car',
    'vehicle.lincoln.mkz2017': 'car',
    'vehicle.nissan.370z': 'car',
    'vehicle.sheriff.car': 'car',
    'vehicle.audi.a2': 'car',
    'vehicle.citroen.c3': 'car',
    'vehicle.jeep.grandcherokee': 'car',
    'vehicle.mini.clubman': 'car',
    'vehicle.opel.corsa': 'car',
    'vehicle.tesla.model3_future': 'car',
    'vehicle.tesla.cybertruck': 'car',
    'vehicle.volkswagen.t2_2021': 'van',
    'vehicle.mustanggt': 'car',
    'vehicle.ford_escort': 'car',
    'vehicle.carlamotors.european_hatchback': 'car',
    'vehicle.audi.a4': 'car',
    'vehicle.mercedes.coupe_2021': 'car',
    'vehicle.mercedes.coupe_2016': 'car',
    'vehicle.mitsubishi.fuso': 'truck',
    'vehicle.mitsubishi.fusotruck': 'truck',
    'vehicle.mitsubishi.fusotruck03': 'truck',
    'vehicle.carlamotors.coupe': 'car',
    'vehicle.carlamotors.family': 'car',
    'vehicle.dodge.charger': 'car',
    'vehicle.lincoln.mkz2018': 'car',
    'vehicle.mercedes.sprinter_2019': 'van',
    'vehicle.tesla.model3_2019': 'car',
    'vehicle.toyota.prius_2017': 'car',
    'vehicle.nissan.370z_2020': 'car',
    'vehicle.ford.mustang_2020': 'car',
    'vehicle.seat.leon': 'car',
    'vehicle.bmw.grandtourer': 'car',
    'vehicle.carlamotors.cabriolet': 'car',
    'vehicle.kawasaki.ninja': 'motorcycle',
    'vehicle.carlamotors.truck': 'truck',
    'vehicle.ford.flareside': 'car',
    'vehicle.ford.f150': 'truck',
    'vehicle.ford.mustang_2021': 'car',
    'vehicle.tesla.model3_2021': 'car',
    'vehicle.bh.crossbike01': 'bicycle',
    'vehicle.bh.crossbike02': 'bicycle',
    'vehicle.bh.crossbike03': 'bicycle',
    'vehicle.diamondback.century01': 'bicycle',
    'vehicle.diamondback.century02': 'bicycle',
    'vehicle.gazelle.omafiets01': 'bicycle',
    'vehicle.dodge_charger_police': 'car',
    'vehicle.ford.crown_police': 'car',
    'vehicle.ford.crown_police_2020': 'car',
    'vehicle.carlamotors.firetruck02': 'truck',
    'vehicle.carlamotors.firetruck03': 'truck',
    'vehicle.volkswagen.t2_2021_cargo': 'van',
    'vehicle.mitsubishi.outlander': 'car',
    'vehicle.carlamotors.polaris': 'car',
    'vehicle.nissan.patrol': 'car',
    'vehicle.toyota.prius_2021': 'car',
    'vehicle.lincoln.mkz_2021': 'car',
    'vehicle.vehicle': 'others',
    'vehicle.dodge.police': 'car',
    'vehicle.gmc.savana': 'van',
    'vehicle.gmc': 'car',
    'vehicle.tesla.model3_2018': 'car',
    'vehicle.audi': 'car',
    'vehicle.mercedes.sprinter_2020': 'van',
    'vehicle.ford.f150_2020': 'truck',
    'vehicle.dodge.charger_police_2021': 'car',
    'vehicle.nissan': 'car',
    'vehicle.carlamotors.european_hatchback_2020': 'car',
    'vehicle.bmw': 'car',
    'vehicle.bmw.330': 'car',
    'vehicle.bmw.grandtourer_2021': 'car',
    'vehicle.carlamotors.european_hatchback_2021': 'car',
    'vehicle.chevrolet': 'car',
    'vehicle.chevrolet.impala_2020': 'car',
    'vehicle.chevrolet.impala_2021': 'car',
    'vehicle.lincoln': 'car',
    'vehicle.lincoln.mkz_2021_police': 'car',
    'vehicle.mercedes': 'car',
    'vehicle.mercedes.coupe_2021_police': 'car',
    'vehicle.mercedes.coupe_2020_police': 'car',
    'vehicle.peterbilt.pull': 'truck',
    'vehicle.peterbilt.tow': 'truck',
    'vehicle.pontiac.firebird': 'car',
    'vehicle.toyota.prius_2020': 'car',
    'vehicle.toyota': 'car',
    'vehicle.toyota.prius_2022': 'car',
    'vehicle.volkswagen.t2_2020': 'van',
    'vehicle.gazelle.omafiets03': 'bicycle',
    'vehicle.gazelle.omafiets02': 'bicycle',
    'vehicle.diamondback.century03': 'bicycle',
    'vehicle.bh.crossbike04': 'bicycle',
    'vehicle.diamondback.century04': 'bicycle',
    'vehicle.carlamotors.cab': 'car',
    'vehicle.mini.cooper_s_2018': 'car',
    'vehicle.buggy': 'car',
    'vehicle.dodge.city': 'car',
    'vehicle.volkswagen.t2_2019': 'van',
    'vehicle.chevrolet.impala_1971': 'car',
    'vehicle.lincoln.mkz_2022': 'car',
    'vehicle.lincoln.mkz_2019': 'car',
    'vehicle.mercedes.coupe_2018': 'car',
    'vehicle.nissan.patrol_2019': 'car',
    'vehicle.tesla.model3_2017': 'car',
    'vehicle.tesla.model3_2016': 'car',
    'vehicle.toyota.prius_2019': 'car',
    'vehicle.ford.f150_2019': 'truck',
    'vehicle.gmc.pickup_2019': 'truck',
    'vehicle.carlamotors.european_hatchback_2019': 'car',
    'vehicle.carlamotors.firetruck01': 'truck',
    'vehicle.volkswagen.t2_2018': 'van',
    'vehicle.carlamotors.firetruck04': 'truck',
    'vehicle.peterbilt.pull_2020': 'truck',
    'vehicle.peterbilt.tow_2020': 'truck',
    'vehicle.dodge.pickup_2020': 'truck',
    'vehicle.carlamotors.coupe_2019': 'car',
    'vehicle.carlamotors.coupe_2020': 'car',
    'vehicle.carlamotors.coupe_2021': 'car',
    'vehicle.carlamotors.family_2019': 'car',
    'vehicle.carlamotors.family_2020': 'car',
    'vehicle.carlamotors.family_2021': 'car',
    'vehicle.carlamotors.cabriolet_2019': 'car',
    'vehicle.carlamotors.cabriolet_2020': 'car',
    'vehicle.carlamotors.cabriolet_2021': 'car',
    'vehicle.carlamotors.coupe_2018': 'car',
    'vehicle.carlamotors.family_2018': 'car',
    'vehicle.carlamotors.cabriolet_2018': 'car',
    'vehicle.carlamotors.coupe_police': 'car',
    'vehicle.carlamotors.family_police': 'car',
    'vehicle.mercedes.coupe_police': 'car',
    'vehicle.dodge.charger_police_2018': 'car',
    'vehicle.dodge.charger_police_2019': 'car',
    'vehicle.dodge.charger_police_2022': 'car',
    'vehicle.carlamotors.cab_2020': 'car',
    'vehicle.carlamotors.cab_2021': 'car',
    'vehicle.carlamotors.truck_2018': 'truck',
    'vehicle.carlamotors.truck_2019': 'truck',
    'vehicle.carlamotors.truck_2020': 'truck',
    'vehicle.carlamotors.truck_2021': 'truck',
    'vehicle.tesla.cybertruck_2020': 'truck',
    'vehicle.tesla.cybertruck_2021': 'truck',
    'vehicle.tesla.cybertruck_2022': 'truck',
    'vehicle.carlamotors.polaris_2020': 'car',
    'vehicle.carlamotors.polaris_2021': 'car',
    'vehicle.carlamotors.polaris_2022': 'car',
    'vehicle.chevrolet.impala_2018': 'car',
    'vehicle.chevrolet.impala_2019': 'car',
    'vehicle.volkswagen.t2_2022': 'van',
    'vehicle.carlamotors.european_hatchback_2022': 'car',
    'vehicle.dodge.charger_2018': 'car',
    'vehicle.dodge.charger_2019': 'car',
    'vehicle.dohany': 'others',
    'vehicle.ford.mustang_2019': 'car',
    'vehicle.nissan.patrol_2022': 'car',
    'vehicle.ford.f150_2022': 'truck',
    'vehicle.ford.f150_2021': 'truck',
    'vehicle.gmc.pickup_2022': 'truck',
    'vehicle.gmc.pickup_2021': 'truck',
    'vehicle.tesla.model3_2022': 'car',
    'vehicle.lotus.elise': 'car',
    'vehicle.carlamotors.european_hatchback_2017': 'car',
    'vehicle.carlamotors.family_2017': 'car',
    'vehicle.carlamotors.coupe_2017': 'car',
    'vehicle.carlamotors.cabriolet_2017': 'car',
    'vehicle.tesla.model3_future2': 'car',
    'vehicle.tesla.model3_future3': 'car',
    'vehicle.carlamotors.cab_2019': 'car',
    'vehicle.carlamotors.cab_2018': 'car',
    'vehicle.carlamotors.cab_2017': 'car',
    'vehicle.carlamotors.truck_2017': 'truck',
    'vehicle.carlamotors.truck_2016': 'truck',
    'vehicle.carlamotors.cabriolet_police': 'car',
    'vehicle.mercedes.coupe_2017': 'car',
    'vehicle.mercedes.coupe_2016_police': 'car',
    'vehicle.sheriff.car_2020': 'car',
    'vehicle.sheriff.car_2021': 'car',
    'vehicle.dodge.charger_police_2017': 'car',
    'vehicle.dodge.charger_2017': 'car',
    'vehicle.volkswagen.t2_2017': 'van',
    'vehicle.volkswagen.t2_2016': 'van',
    'vehicle.ford.f150_2018': 'truck',
    'vehicle.tesla.model3_2015': 'car',
    'vehicle.chevrolet.impala_2017': 'car',
    'vehicle.gmc.pickup_2018': 'truck',
    'vehicle.dodge.pickup_2019': 'truck',
    'vehicle.dodge.pickup_2018': 'truck',
    'vehicle.dodge.pickup_2017': 'truck',
    'vehicle.mustang': 'car',
    'vehicle.mustang_2021': 'car',
    'vehicle.mustang_2022': 'car',
    'vehicle.mustang_2023': 'car',
    'vehicle.mustang.gt': 'car',
    'vehicle.volkswagen.t2_cargo': 'van',
    'vehicle.tesla.cybertruck_2023': 'truck',
    'vehicle.ford.f150_2023': 'truck',
    'vehicle.gmc.pickup_2023': 'truck',
    'vehicle.dodge.pickup_2023': 'truck',
    'vehicle.tesla.cybertruck_2021': 'truck',
    'vehicle.tesla.cybertruck_2024': 'truck',
    'vehicle.tesla.cybertruck_2025': 'truck',
    'vehicle.dodge.pickup_2024': 'truck',
    'vehicle.carlamotors.truck_2022': 'truck',
    'vehicle.carlamotors.coupe_2022': 'car',
    'vehicle.carlamotors.cabriolet_2022': 'car',
    'vehicle.carlamotors.family_2022': 'car',
    'vehicle.carlamotors.european_hatchback_2023': 'car',
    'vehicle.carlamotors.coupe_2023': 'car',
    'vehicle.carlamotors.cabriolet_2023': 'car',
    'vehicle.carlamotors.family_2023': 'car',
    'vehicle.carlamotors.european_hatchback_2024': 'car',
    'vehicle.carlamotors.coupe_2024': 'car',
    'vehicle.carlamotors.cabriolet_2024': 'car',
    'vehicle.carlamotors.family_2024': 'car',
    'vehicle.carlamotors.european_hatchback_2025': 'car',
    'vehicle.carlamotors.coupe_2025': 'car',
    'vehicle.carlamotors.cabriolet_2025': 'car',
    'vehicle.carlamotors.family_2025': 'car',
    'vehicle.carlamotors.european_hatchback_2015': 'car',
    'vehicle.carlamotors.family_2015': 'car',
    'vehicle.carlamotors.coupe_2015': 'car',
    'vehicle.carlamotors.cabriolet_2015': 'car',
    'vehicle.carlamotors.truck_2015': 'truck',
    'vehicle.carlamotors.coupe_2014': 'car',
    'vehicle.carlamotors.family_2014': 'car',
    'vehicle.carlamotors.cabriolet_2014': 'car',
    'vehicle.carlamotors.truck_2014': 'truck',
    'vehicle.carlamotors.coupe_2013': 'car',
    'vehicle.carlamotors.family_2013': 'car',
    'vehicle.carlamotors.cabriolet_2013': 'car',
    'vehicle.carlamotors.truck_2013': 'truck',
    'vehicle.carlamotors.coupe_2012': 'car',
    'vehicle.carlamotors.family_2012': 'car',
    'vehicle.carlamotors.cabriolet_2012': 'car',
    'vehicle.carlamotors.truck_2012': 'truck',
    'vehicle.carlamotors.coupe_2011': 'car',
    'vehicle.carlamotors.family_2011': 'car',
    'vehicle.carlamotors.cabriolet_2011': 'car',
    'vehicle.carlamotors.truck_2011': 'truck',
    'vehicle.carlamotors.coupe_2010': 'car',
    'vehicle.carlamotors.family_2010': 'car',
    'vehicle.carlamotors.cabriolet_2010': 'car',
    'vehicle.carlamotors.truck_2010': 'truck',
    'vehicle.carlamotors.coupe_2009': 'car',
    'vehicle.carlamotors.family_2009': 'car',
    'vehicle.carlamotors.cabriolet_2009': 'car',
    'vehicle.carlamotors.truck_2009': 'truck',
    'vehicle.carlamotors.coupe_2008': 'car',
    'vehicle.carlamotors.family_2008': 'car',
    'vehicle.carlamotors.cabriolet_2008': 'car',
    'vehicle.carlamotors.truck_2008': 'truck',
    'vehicle.carlamotors.coupe_2007': 'car',
    'vehicle.carlamotors.family_2007': 'car',
    'vehicle.carlamotors.cabriolet_2007': 'car',
    'vehicle.carlamotors.truck_2007': 'truck',
    'vehicle.carlamotors.coupe_2006': 'car',
    'vehicle.carlamotors.family_2006': 'car',
    'vehicle.carlamotors.cabriolet_2006': 'car',
    'vehicle.carlamotors.truck_2006': 'truck',
    'vehicle.carlamotors.coupe_2005': 'car',
    'vehicle.carlamotors.family_2005': 'car',
    'vehicle.carlamotors.cabriolet_2005': 'car',
    'vehicle.carlamotors.truck_2005': 'truck',
    'vehicle.carlamotors.coupe_2004': 'car',
    'vehicle.carlamotors.family_2004': 'car',
    'vehicle.carlamotors.cabriolet_2004': 'car',
    'vehicle.carlamotors.truck_2004': 'truck',
    'vehicle.carlamotors.coupe_2003': 'car',
    'vehicle.carlamotors.family_2003': 'car',
    'vehicle.mitsubishi.fusotruck02': 'truck',
    'vehicle.gazelle.omafiets04': 'bicycle',
    'vehicle.diamondback.century05': 'bicycle',
    'vehicle.bh.crossbike05': 'bicycle',
    'vehicle.bh.crossbike06': 'bicycle',
    'vehicle.diamondback.century06': 'bicycle',
    'vehicle.gazelle.omafiets05': 'bicycle',
    'vehicle.gazelle.omafiets06': 'bicycle',
    'vehicle.bh.crossbike07': 'bicycle',
    'vehicle.diamondback.century07': 'bicycle',
    'vehicle.nissan.nissan_dolvo': 'car',
    'vehicle.carlamotors.bus': 'bus',
    'vehicle.mercedes.bus': 'bus',
    'vehicle.toyota.prius_xlight': 'car',
    'vehicle.mercedes.sprinter_xlight': 'van',
    'vehicle.tesla.model3_xlight': 'car',
    'vehicle.ford.mustang_xlight': 'car',
    'vehicle.lincoln.mkz_xlight': 'car',
    'vehicle.volkswagen.t2_xlight': 'van',
    'vehicle.ford.f150_xlight': 'truck',
    'vehicle.gmc.pickup_xlight': 'truck',
    'vehicle.bh.crossbike_xlight': 'bicycle',
    'vehicle.diamondback.century_xlight': 'bicycle',
    'vehicle.gazelle.omafiets_xlight': 'bicycle',
    'vehicle.mercedes.coupe_xlight': 'car',
    'vehicle.dodge.charger_xlight': 'car',
    'vehicle.dodge.charger_police_xlight': 'car',
    'vehicle.carlamotors.coupe_xlight': 'car',
    'vehicle.carlamotors.family_xlight': 'car',
    'vehicle.carlamotors.cabriolet_xlight': 'car',
    'vehicle.carlamotors.truck_xlight': 'truck',
    'vehicle.mitsubishi.fuso_xlight': 'truck',
    'vehicle.carlamotors.firetruck_xlight': 'truck',
    'vehicle.carlamotors.cab_xlight': 'car',
    'vehicle.carlamotors.polaris_xlight': 'car',
    'vehicle.tesla.cybertruck_xlight': 'truck',
    'vehicle.peterbilt.pull_xlight': 'truck',
    'vehicle.peterbilt.tow_xlight': 'truck',
    'vehicle.dodge.pickup_xlight': 'truck',
    'vehicle.ford.ambulance_xlight': 'van',
    'vehicle.carlamotors.firetruck01_xlight': 'truck',
    'vehicle.carlamotors.firetruck02_xlight': 'truck',
    'vehicle.carlamotors.firetruck03_xlight': 'truck',
    'vehicle.volkswagen.t2_2021_xlight': 'van',
    'vehicle.carlamotors.european_hatchback_xlight': 'car',
    'vehicle.bmw.grandtourer_xlight': 'car',
    'vehicle.nissan.370z_xlight': 'car',
    'vehicle.seat.leon_xlight': 'car',
    'vehicle.chevrolet.impala_xlight': 'car',
    'vehicle.lincoln.mkz2017_xlight': 'car',
    'vehicle.citroen.c3_xlight': 'car',
    'vehicle.mini.clubman_xlight': 'car',
    'vehicle.audi.a2_xlight': 'car',
    'vehicle.bmw.130_xlight': 'car',
    'vehicle.mercedes.coupe_2020_xlight': 'car',
    'vehicle.opel.corsa_xlight': 'car',
    'vehicle.jeep.grandcherokee_xlight': 'car',
    'vehicle.jeep.wrangler_rubicon_xlight': 'car',
    'vehicle.audi.a4_xlight': 'car',
    'vehicle.toyota.prius_2021_xlight': 'car',
    'vehicle.mercedes.coupe_2021_xlight': 'car',
    'vehicle.nissan.370z_2020_xlight': 'car',
    'vehicle.ford.mustang_2020_xlight': 'car',
    'vehicle.carlamotors.cab_2020_xlight': 'car',
    'vehicle.carlamotors.cab_2021_xlight': 'car'
})

class_names = [
'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
]

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

# 🔥 关键：最小化配置  TODO 此处相较原始有更改
num_gpus = 1
batch_size = 1
num_iters_per_epoch = 3  # 只运行3个iter测试流程
num_epochs = 1  # 只训练1个epoch

llm_path = 'ckpts/pretrain_qformer/'
use_gen_token = False
collect_keys = ['lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command']
pretrain = True
use_col_loss = False



# 模型配置
model = dict(
    type='Orion',
    save_path='./results_planning_only/',
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    # 🔥 关键修改：禁用LLM加载（防止内存溢出死机）
    # LLM模型权重10-20GB，加载时会瞬间占用大量内存
    # 对于流程测试，可以先不加载LLM
    tokenizer=llm_path,  # 原值：llm_path
    lm_head=None,    # 原值：llm_path
    use_gen_token=False,  # 确保为False
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


# 数据配置
dataset_type = 'B2DOrionDataset'
data_root = 'data/bench2drive'
info_root = 'data/infos'
map_root = 'data/bench2drive/maps'
map_file = 'data/infos/b2d_map_infos.pkl'

file_client_args = dict(backend="disk")
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
    # 🔥 暂时注释掉VQA加载（需要tokenizer）   TODO
    # dict(
    #     type='LoadAnnoatationVQA',
    #     base_desc_path='./data/chat-B2D/train',
    #     tokenizer=llm_path,
    #     max_length=2048,
    #     use_gen_token=use_gen_token,
    #     pretrain=pretrain),
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
            # 'input_ids', 'vlm_labels',  # VQA相关，已移除
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
    # 🔥 暂时注释掉VQA加载（需要tokenizer）
    # dict(
    #     type='LoadAnnoatationCriticalVQATest',
    #     load_type=['critical_qa'],
    #     tokenizer=llm_path,
    #     use_gen_token=use_gen_token,
    #     max_length=2048),
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
                    # 'input_ids', 'vlm_labels',  # VQA相关，已移除
                    'gt_attr_labels', 'ego_fut_trajs',
                    'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                    'can_bus', 'fut_valid_flag', 'lidar2img',
                    'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'command'
                ])
        ])
]
inference_only_pipeline = [
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args, img_root=data_root),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D", keys=[
                                            "img",
                                            "timestamp",
                                            "l2g_r_mat",
                                            "l2g_t",
                                            "command",
                                        ]
            ),
        ],
    ),
]
# 🔥 数据集配置 - 极少量样本
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,  # 单进程加载
    train=dict(
        type='B2DOrionDataset',
        data_root=data_root,
        ann_file=ann_file_train,
        limit_samples=8,  # ⚠️ 只加载5个训练样本
        pipeline=train_pipeline,
        classes=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        seq_mode=False,  # 关闭序列模式，避免小数据集分组报错
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
        limit_samples=6,  # ⚠️ 只加载5个验证样本（避免数据集为空）
        pipeline=test_pipeline,
        classes=['car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone', 'traffic_light', 'pedestrian', 'others'],
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        seq_mode=False,
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
    # 采样器配置（IterBasedRunner 推荐使用“无限”批采样器，避免小样本触发 StopIteration）
    # 训练：使用 InfiniteGroupEachSampleInBatchSampler（与 IterBasedRunner 配合）
    # 验证：使用常规 DistributedSampler
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=1,
        warmup_split_num=0,
        num_iters_to_seq=3  # 与 total_iters 保持一致：num_iters_per_epoch(3) * num_epochs(1)
    ),
    nonshuffler_sampler=dict(type='DistributedSampler', shuffle=False)
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

# 当 cfg.fp16 存在时，框架会在代码中用 Fp16OptimizerHook(**cfg.optimizer_config, **fp16)
# 因此这里不要再提供 type / loss_scale，避免重复关键字冲突
optimizer_config = dict(
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
evaluation = dict(interval=999999, pipeline=None)  # 暂时跳过评估，避免验证数据集为空的问题

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
print("⚠️  最小化训练配置已加载（流程测试模式）")
print(f"训练样本数: 5")
print(f"验证样本数: 3")
print(f"迭代次数: {total_iters}")
print(f"Batch Size: {batch_size}")
print("🔥 LLM加载: 已禁用（防止内存溢出）")
print("   注意：此配置仅用于测试流程，不包含VQA功能")
print("=" * 80)

