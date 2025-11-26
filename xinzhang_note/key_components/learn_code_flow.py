#!/usr/bin/env python3
"""
代码流程学习脚本 - 逐步展示 Orion 项目的代码调用链
无需真实数据，用打印展示代码执行流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_step(step_num, title, details=""):
    """打印步骤信息"""
    print(f"\n{'='*80}")
    print(f"步骤 {step_num}: {title}")
    print(f"{'='*80}")
    if details:
        print(details)
    print()

def explore_training_pipeline():
    """探索训练流程"""
    
    print("\n" + "🎓 Orion 代码流程学习".center(80, "="))
    print("\n本脚本展示 Orion 项目的代码调用链，无需真实数据\n")
    
    # ==================== 步骤 1 ====================
    print_step(1, "训练入口: train.py", 
               "文件位置: adzoo/orion/train.py\n"
               "启动命令: python train.py <config_file>")
    
    print("📄 train.py 的主要功能:")
    print("  1. 解析命令行参数")
    print("  2. 加载配置文件 (Config.fromfile)")
    print("  3. 构建模型 (build_model)")
    print("  4. 构建数据集 (build_dataset)")
    print("  5. 调用训练函数 (custom_train_model)")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 2 ====================
    print_step(2, "配置文件解析",
               "文件位置: adzoo/orion/configs/orion_stage1_train.py")
    
    print("📋 配置文件定义:")
    print("  - model: 模型结构配置")
    print("    ├─ type='Orion'  → 使用 Orion 类")
    print("    ├─ img_backbone   → EVAViT backbone")
    print("    ├─ pts_bbox_head  → OrionHead (检测)")
    print("    ├─ map_head       → OrionHeadM (地图)")
    print("    └─ lm_head        → LLM head (规划)")
    print()
    print("  - data: 数据集配置")
    print("  - optimizer: 优化器配置")
    print("  - lr_config: 学习率调度配置")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 3 ====================
    print_step(3, "模型构建: build_model()",
               "代码位置: train.py 第196-199行")
    
    print("🏗️  build_model 流程:")
    print()
    print("  from mmcv.models import builder")
    print("  model = builder.build_detector(cfg.model)")
    print()
    print("  ↓")
    print("  根据 cfg.model['type']='Orion' 查找对应的类")
    print("  ↓")
    print("  调用 Orion.__init__() 构造函数")
    print("  ↓")
    print("  Orion 继承自 MVXTwoStageDetector")
    print("  ↓")
    print("  在父类 __init__ 中构建各个子模块:")
    print("    • img_backbone = builder.build_backbone(img_backbone)")
    print("    • pts_bbox_head = builder.build_head(pts_bbox_head)")
    print("    • map_head = builder.build_head(map_head)")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 4 ====================
    print_step(4, "Orion 类结构",
               "代码位置: mmcv/models/detectors/orion.py")
    
    print("📦 Orion 类层次:")
    print()
    print("  Orion (第67行)")
    print("    ↓ 继承")
    print("  MVXTwoStageDetector (mmcv/models/detectors/mvx_two_stage.py)")
    print("    ↓ 继承")
    print("  Base3DDetector")
    print()
    print("🔧 Orion 类的关键属性:")
    print("  - self.img_backbone: 图像特征提取 (EVAViT)")
    print("  - self.pts_bbox_head: 3D目标检测头 (OrionHead)")
    print("  - self.map_head: 地图元素检测头 (OrionHeadM)")
    print("  - self.lm_head: 视觉语言模型头 (LLaMA)")
    print()
    print("🔄 Orion 类的关键方法:")
    print("  - __init__(): 初始化各模块")
    print("  - forward(): 前向传播入口")
    print("  - forward_train(): 训练时的前向传播")
    print("  - forward_pts_train(): 点云/检测分支的训练")
    print("  - extract_feat(): 提取图像特征")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 5 ====================
    print_step(5, "OrionHead 实例创建",
               "代码位置: mmcv/models/detectors/mvx_two_stage.py 第59-64行")
    
    print("🎯 OrionHead 创建流程:")
    print()
    print("  在 MVXTwoStageDetector.__init__() 中:")
    print()
    print("  ```python")
    print("  if pts_bbox_head:")
    print("      pts_train_cfg = train_cfg.pts if train_cfg else None")
    print("      pts_bbox_head.update(train_cfg=pts_train_cfg)")
    print("      pts_test_cfg = test_cfg.pts if test_cfg else None")
    print("      pts_bbox_head.update(test_cfg=pts_test_cfg)")
    print("      self.pts_bbox_head = builder.build_head(pts_bbox_head)  # ← 创建实例")
    print("  ```")
    print()
    print("  ↓")
    print("  builder.build_head() 根据 type='OrionHead' 创建 OrionHead 实例")
    print("  ↓")
    print("  OrionHead.__init__() 被调用，初始化:")
    print("    • Query embeddings (600个)")
    print("    • Transformer layers")
    print("    • 分类和回归头")
    print("    • 损失函数")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 6 ====================
    print_step(6, "训练前向传播流程",
               "数据流: 图像 → 特征 → 检测 → VLM → 规划")
    
    print("🔄 完整的前向传播调用链:")
    print()
    print("  1. train.py → custom_train_model()")
    print("     ↓")
    print("  2. 训练循环: for epoch in range(epochs):")
    print("     ↓")
    print("  3. model.forward(data, return_loss=True)")
    print("     ↓  (orion.py 第421行)")
    print("  4. model.forward_train(**data)")
    print("     ↓  (orion.py 第441行)")
    print("  5. 提取图像特征:")
    print("     data['img_feats'] = model.extract_feat(data['img'])")
    print("     ↓  (orion.py 第362行)")
    print("  6. model.forward_pts_train()")
    print("     ↓  (orion.py 第508行)")
    print("  7. 准备位置编码:")
    print("     location = model.prepare_location()")
    print("     pos_embed = model.position_embeding()")
    print("     ↓")
    print("  8. 检测头处理:")
    print("     outs_bbox, det_query = model.pts_bbox_head(pos_embed, **data)")
    print("     ↓  (OrionHead forward)")
    print("  9. 地图头处理:")
    print("     outs_lane, map_query = model.map_head(pos_embed, **data)")
    print("     ↓  (OrionHeadM forward)")
    print(" 10. VLM处理:")
    print("     vision_embeded = torch.cat([det_query, map_query])")
    print("     vlm_loss, ego_feature = model.lm_head(input_ids, vision_embeded)")
    print("     ↓")
    print(" 11. 规划解码:")
    print("     ego_fut_preds = model.ego_fut_decoder(ego_feature)")
    print("     ↓")
    print(" 12. 损失计算和反向传播")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 7 ====================
    print_step(7, "OrionHead.forward() 详细流程",
               "代码位置: mmcv/models/dense_heads/orion_head.py")
    
    print("🎯 OrionHead.forward() 内部流程:")
    print()
    print("  输入: pos_embed (位置编码), img_metas, **data")
    print("  ↓")
    print("  1. 初始化 query embeddings (可学习参数)")
    print("  ↓")
    print("  2. 如果使用 memory:")
    print("     从历史帧获取 memory queries")
    print("  ↓")
    print("  3. 通过 Transformer 解码:")
    print("     query = self.transformer(")
    print("         query=query,")
    print("         key=img_features,")
    print("         value=img_features,")
    print("         pos_embed=pos_embed")
    print("     )")
    print("  ↓")
    print("  4. 预测头:")
    print("     cls_scores = self.cls_branches(query)")
    print("     bbox_preds = self.reg_branches(query)")
    print("  ↓")
    print("  5. 输出:")
    print("     return outs, det_query")
    print("       ├─ outs: 包含 cls_scores, bbox_preds 等")
    print("       └─ det_query: 用于后续 VLM 的特征 (B, 600, 4096)")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 8 ====================
    print_step(8, "关键数据结构",
               "各模块之间传递的数据格式")
    
    print("📊 主要数据结构:")
    print()
    print("1️⃣  图像数据:")
    print("   data['img']: torch.Tensor [B, N_cam, 3, H, W]")
    print("   └─ B=1, N_cam=6, H=640, W=640")
    print()
    print("2️⃣  图像特征:")
    print("   img_feats: torch.Tensor [B, N_cam, C, H', W']")
    print("   └─ C=1024 (EVAViT embed_dim)")
    print()
    print("3️⃣  位置编码:")
    print("   pos_embed: torch.Tensor [B, N_cam*H'*W', 256]")
    print("   └─ 将所有相机的特征展平后的位置编码")
    print()
    print("4️⃣  检测 query:")
    print("   det_query: torch.Tensor [B, 600, 4096]")
    print("   └─ 600 个检测 query，每个 4096 维")
    print()
    print("5️⃣  地图 query:")
    print("   map_query: torch.Tensor [B, 600, 4096]")
    print("   └─ 600 个地图 query，每个 4096 维")
    print()
    print("6️⃣  VLM 输入:")
    print("   vision_embeded: torch.Tensor [B, 1200, 4096]")
    print("   └─ 拼接 det_query 和 map_query")
    print()
    print("7️⃣  规划特征:")
    print("   ego_feature: torch.Tensor [B, 4096]")
    print("   └─ 从 LLM 提取的规划 token")
    
    input("\n按 Enter 继续...")
    
    # ==================== 步骤 9 ====================
    print_step(9, "损失计算",
               "各模块的损失函数")
    
    print("📉 损失项:")
    print()
    print("1️⃣  检测损失 (OrionHead.loss):")
    print("   • loss_cls: 分类损失 (Focal Loss)")
    print("   • loss_bbox: 边界框回归损失 (L1 Loss)")
    print("   • loss_iou: IoU 损失 (GIoU Loss)")
    print()
    print("2️⃣  地图损失 (OrionHeadM.loss):")
    print("   • loss_cls: 车道线分类损失")
    print("   • loss_bbox: 车道线点回归损失")
    print()
    print("3️⃣  VLM 损失:")
    print("   • vlm_loss: 语言模型损失 (Cross Entropy)")
    print()
    print("4️⃣  规划损失:")
    print("   • loss_plan_reg: 轨迹回归损失 (L1 Loss)")
    print("   • loss_plan_bound: 边界约束损失")
    print("   • loss_vae_gen: VAE 生成损失 (KL散度)")
    print()
    print("💡 总损失 = 各项损失的加权和")
    
    input("\n按 Enter 继续...")
    
    # ==================== 总结 ====================
    print("\n" + "="*80)
    print("✅ 代码流程学习完成！".center(80))
    print("="*80 + "\n")
    
    print("📚 学习建议:")
    print()
    print("1. 从简单到复杂:")
    print("   • 先理解 Orion 的整体结构")
    print("   • 再深入每个子模块 (OrionHead, OrionHeadM)")
    print("   • 最后研究 VLM 和规划模块")
    print()
    print("2. 添加调试打印:")
    print("   • 在 orion.py 的 forward_pts_train() 添加打印")
    print("   • 在 orion_head.py 的 forward() 添加打印")
    print("   • 观察数据的 shape 和数值范围")
    print()
    print("3. 使用测试脚本:")
    print("   • 运行: python test_model_flow.py")
    print("   • 无需真实数据，快速测试模型结构")
    print()
    print("4. 关键文件:")
    print("   • mmcv/models/detectors/orion.py       - 主模型")
    print("   • mmcv/models/dense_heads/orion_head.py - 检测头")
    print("   • mmcv/models/dense_heads/orion_head_map.py - 地图头")
    print("   • adzoo/orion/train.py                 - 训练入口")
    print()
    print("💡 提示: 如果要真正运行训练，需要:")
    print("   1. 准备数据集 (data/bench2drive/)")
    print("   2. 下载预训练权重 (ckpts/)")
    print("   3. 配置 tokenizer (ckpts/pretrain_qformer/)")
    print()

if __name__ == "__main__":
    try:
        explore_training_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  学习中断\n")

