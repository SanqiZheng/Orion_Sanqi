
def forward_pts_train(self,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        gt_attr_labels,
                        map_gt_bboxes_3d,
                        map_gt_labels_3d,   
                        img_metas,
                        input_ids, 
                        vlm_labels, 
                        vlm_attn_mask,
                        ego_fut_trajs,
                        **data):
    """Forward function for point cloud branch.
    Args:
        pts_feats (list[torch.Tensor]): Features of point cloud branch
        gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
            boxes for each sample.
        gt_labels_3d (list[torch.Tensor]): Ground truth labels for
            boxes of each sampole
        img_metas (list[dict]): Meta information of samples.
        gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
            boxes to be ignored. Defaults to None.
    Returns:
        dict: Losses of each branch.
    """
    B = data['img'].shape[0]
    location = self.prepare_location(img_metas, **data) # (6, 40, 40, 2)
    pos_embed = self.position_embeding(data, location, img_metas) # (1, 9600, 256)
    losses = dict()

    if self.with_pts_bbox:
        outs_bbox, det_query = self.pts_bbox_head(img_metas, pos_embed, **data) # (1, 257, 4096)
        vision_embeded_obj = det_query.clone()
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs_bbox, gt_attr_labels]
        if self.pts_bbox_head.pred_traffic_light_state:
            loss_inputs.append(data['traffic_state'])
            loss_inputs.append(data['traffic_state_mask'])
        if self.use_col_loss:
            loss, agent_outs = self.pts_bbox_head.loss(*loss_inputs)
        else:
            loss = self.pts_bbox_head.loss(*loss_inputs)
        losses.update(loss)
    
    """
    BEV 提取为query, 在 forward_pts_train() 中与文本输入一起送入 VLM (LLM + 视觉投影), 随后用生成式规划头把LLM的规划
    token 解码为轨迹， 实现 "视觉 -> 语言推理 -> 运动规划" 的链路
    
    """
    if self.with_map_head:
        outs_lane, map_query = self.map_head(img_metas, pos_embed, **data)
        vision_embeded_map = map_query.clone()
        # reference vad trans
        device = gt_labels_3d[0].device
        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_3d)
        lane_pts = [F.pad(map_gt_bboxes.fixed_num_sampled_points.to(device),(0,1)) for map_gt_bboxes in map_gt_vecs_list]
        loss_inputs = [lane_pts, map_gt_labels_3d, outs_lane, img_metas]

        if False:
            # for debug
            import pickle
            with open('lane_pts.pkl', 'wb') as file:
                pickle.dump(lane_pts, file)
         # 训练/推理时都在同一模块内串联 VLM loss 与规划 loss，使感知、语言推理与控制可以端到端联合优化
        losses.update(self.map_head.loss(*loss_inputs))        

    if self.with_lm_head:
        if self.use_gen_token:
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
            vlm_loss, ego_feature = self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False, return_ego_feature=True)
            if self.mix_qa_training:
                dummy_ego_feature = self.lm_head.get_model().embed_tokens(torch.tensor([[self.lm_head.config.waypoint_token_idx] for _ in range(B)]).cuda())
                dummy_ego_feature = dummy_ego_feature.squeeze(1)
                valid_input_mask = (input_ids == self.lm_head.config.waypoint_token_idx).sum(dim=-1).to(torch.bool)
                dummy_ego_feature[valid_input_mask] = ego_feature
                ego_feature = dummy_ego_feature
                data['ego_fut_masks'][:,0,0] *= valid_input_mask.unsqueeze(-1)
            losses.update(vlm_loss=vlm_loss[0])
            current_states = ego_feature.unsqueeze(1)

            if not self.use_diff_decoder and not self.use_mlp_decoder:
                distribution_comp = {}
                noise = None
                self.fut_ts = 6
                if self.training:
                    future_distribution_inputs = ego_fut_trajs.reshape(B, ego_fut_trajs.shape[1], -1)
                if self.PROBABILISTIC:
                    sample, output_distribution = self.distribution_forward(
                        current_states, future_distribution_inputs, noise
                    )
                    distribution_comp = {**distribution_comp, **output_distribution}

                hidden_states = ego_feature.unsqueeze(1)
                states_hs, future_states_hs = \
                    self.future_states_predict(B, sample, hidden_states, current_states)

                ego_query_hs = \
                    states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
                ego_fut_trajs_list = []
                for i in range(self.fut_ts):
                    outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
                    ego_fut_trajs_list.append(outputs_ego_trajs)

                ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)
                lane_scores = outs_lane['all_lane_cls_one2one'][-1]
                lane_preds = outs_lane['all_lane_preds_one2one'][-1]
                for p in range(self.map_head.n_control):
                    lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                    lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
                if self.with_bound_loss:
                    loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['ego_fut_cmd'][:,0,0], lane_preds, lane_scores]
                else:
                    loss_plan_input = [ego_fut_preds, ego_fut_trajs[:,0], data['ego_fut_masks'][:,0,0], data['ego_fut_cmd'][:,0,0]]
                
                if self.use_col_loss:
                    loss_planning_dict = self.loss_planning(*loss_plan_input, **agent_outs)
                else:
                    loss_planning_dict = self.loss_planning(*loss_plan_input)
                losses.update(loss_planning_dict)
                loss_vae_gen = self.loss_vae_gen(distribution_comp, data['ego_fut_masks'][:,0,0])
                loss_vae_gen = torch.nan_to_num(loss_vae_gen)
                losses.update(loss_vae_gen=loss_vae_gen)
            elif self.use_diff_decoder:
                bs = B
                device = ego_feature.device
                # 1. add truncated noise to the plan anchor
                plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
                odo_info_fut = self.norm_odo(plan_anchor)
                timesteps = torch.randint(
                    0, 50,
                    (bs,), device=device
                )
                noise = torch.randn(odo_info_fut.shape, device=device)
                noisy_traj_points = self.diffusion_scheduler.add_noise(
                    original_samples=odo_info_fut,
                    noise=noise,
                    timesteps=timesteps,
                ).float()
                noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
                noisy_traj_points = self.denorm_odo(noisy_traj_points)

                # debug visualization
                # ============================================debug===========================================================
                # self.noising_vis(self,plan_anchor,device)
                # ============================================debug===========================================================
                ego_fut_mode = noisy_traj_points.shape[1]
                # 2. proj noisy_traj_points to the query
                traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=512)
                
                traj_pos_embed = traj_pos_embed.flatten(-2)
                traj_feature = self.plan_anchor_encoder(traj_pos_embed)
                traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
                # 3. embed the timesteps
                time_embed = self.time_mlp(timesteps)
                time_embed = time_embed.view(bs,1,-1)

                # 4. begin the stacked decoder
                poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, current_states, time_embed)
                targets = torch.cumsum(ego_fut_trajs,dim=-2).squeeze(1)
                trajectory_loss_dict = {}

                lane_scores = outs_lane['all_lane_cls_one2one'][-1]
                lane_preds = outs_lane['all_lane_preds_one2one'][-1]
                for p in range(self.map_head.n_control):
                    lane_preds[..., 3 * p].clamp_(min=self.map_head.pc_range[0], max=self.map_head.pc_range[3])
                    lane_preds[..., 3 * p + 1].clamp_(min=self.map_head.pc_range[1], max=self.map_head.pc_range[4])
                lane_preds = lane_preds.reshape(lane_preds.shape[0],lane_preds.shape[1],-1,3)[...,:2]
                for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
                    trajectory_cls_loss, trajectory_reg_loss, trajectory_bound_loss = self.loss_planning_diffusion(poses_reg, poses_cls, targets, plan_anchor, data['ego_fut_masks'][:,0,0],lane_preds, lane_scores)
                    trajectory_loss_dict[f"traj_diff_loss_cls_{idx}"] = trajectory_cls_loss
                    trajectory_loss_dict[f"traj_diff_loss_reg_{idx}"] = trajectory_reg_loss
                    trajectory_loss_dict[f"traj_diff_loss_bound_{idx}"] = trajectory_bound_loss
                    
                losses.update(trajectory_loss_dict)
            elif self.use_mlp_decoder:
                waypoint = self.waypoint_decoder(current_states)
                waypoint = waypoint.reshape(-1,2)
                wp_loss = self.waypoints_loss(waypoint.to(torch.float32), ego_fut_trajs.view(-1, 2).to(torch.float32))
                if 'ego_fut_masks' in data: # ignore invalid fut trajs supervision
                    wp_loss = (wp_loss * data['ego_fut_masks'].view(-1, 1)).mean()
                else:
                    wp_loss = wp_loss.mean()
                wp_loss = torch.nan_to_num(wp_loss)
                losses.update(wp_loss=wp_loss)
        else:
            waypoint = None
            vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
            vlm_loss= self.lm_head(input_ids=input_ids, attention_mask=vlm_attn_mask, labels=vlm_labels, images=vision_embeded, use_cache=False)
            losses.update(vlm_loss=vlm_loss[0])
    return losses




# prepare_inputs_labels_for_multimodal() 把文本里的 <image> 占位符替换为视觉序列，统一 padding/attention mask，
# 因此 scene tokens 与文本 token 共用自注意力空间 
def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
):
    
    if  image_features is None or input_ids.shape[1] == 1:
        # if past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
        #     target_shape = past_key_values[-1][-1].shape[-2] + 1
        #     attention_mask = torch.cat((attention_mask, torch.ones(
        #         (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
        #         dtype=attention_mask.dtype,
        #         device=attention_mask.device
        #     )), dim=1)
        #     position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

    
    if isinstance(image_features,list):
        temp_image_features = []
        for b_id in range(len(image_features[0])):
            for img_id in range(len(image_features)):
                temp_image_features.append(image_features[img_id][b_id])
        image_features = temp_image_features
    else:
        image_features = image_features.reshape(image_features.shape[0], -1, self.hidden_size).to(dtype=self.dtype) # (B, 513, 4096)

    # TODO: image start / end is not implemented here to support pretraining.
    # if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #     raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool() # (B, 76)
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) # (76,)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- TODO: double check
    input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    new_input_ids = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids): # 遍历batch samples
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # 只有一个-200，位置在index 35，估计对应的是句子中image的占位
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] # [-1, 35, 76]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1): # 以image token位置为分界，分割出来句子块
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # [(35,), (40,)]，分块input ids
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]]) # [(35,), (40,)]，分块labels
        split_sizes = [x.shape[0] for x in cur_labels_noim] # [35, 40]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim).to(image_features.device)) # (75,) -> (75, 4096)，得到单词embedding
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # [(35, 4096), (40, 4096)]，分成分块单词embedding
        
        cur_new_input_embeds = []
        cur_new_labels = []
        cur_new_input_ids = []

        for i in range(num_images + 1): # 遍历单词分块，在合适位置，embedding给append入image feature，label给append入IGNORE_INDEX，input id给append入IMAGE_TOKEN_INDEX，得到完整句子组成
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            cur_new_input_ids.append(cur_input_ids_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)
        cur_new_input_ids = torch.cat(cur_new_input_ids)
        # 组成batch
        new_input_embeds.append(cur_new_input_embeds) # [(588, 4096)], 588 = 35+513+40
        new_labels.append(cur_new_labels) # [(588,)]
        new_input_ids.append(cur_new_input_ids) # [(588,)]
    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds) # batch内samples的最大长度，这里想把整个batch都整到一起
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    new_inputs_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels,cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
        cur_len = cur_new_embed.shape[0]

        #padding
        new_input_embeds_padded.append(torch.cat((
            cur_new_embed,
            torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
        ), dim=0))
        if cur_len > 0:
            new_labels_padded[i, :cur_len] = cur_new_labels
            new_inputs_ids_padded[i, :cur_len] = cur_new_input_ids
            attention_mask[i, :cur_len] = True
            position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_inputs_ids_padded





"""

    if self.with_lm_head:
        # 推理时会累计历史问答 history_input_output_id，在检测到 <waypoint_ego> 请求后一次性送入 LLM，
        # 并在 inference_ego() 中根据 special token 位置提取其隐藏状态作为 planning token，同时保证对话历史参与注意力
        history_input_output_id = []        # 累计历史问答
        vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1) # (1, 513, 4096)
        for i, input_ids in enumerate(data['input_ids'][0]):
            input_ids = input_ids.unsqueeze(0)
            special_token_inputs = False
            if not self.qa_pretrain:
                if hasattr(self.lm_head.config,'waypoint_token_idx'):
                    if isinstance(self.lm_head.config.waypoint_token_idx,list):
                        for sptoken in self.lm_head.config.waypoint_token_idx:
                            if sptoken in input_ids:
                                special_token_inputs = True
                                break
                    else:
                        special_token_inputs = self.lm_head.config.waypoint_token_idx in input_ids
            if self.use_gen_token and special_token_inputs: # must be the final round conversation
                history_input_output_id.append(input_ids)
                context_input_ids = torch.cat(history_input_output_id,dim=-1)
                ego_feature = self.lm_head.inference_ego(
                    inputs=context_input_ids,
                    images=vision_embeded,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.75,
                    num_beams=1,
                    max_new_tokens=320,
                    use_cache=True,
                    return_ego_feature=True
                )

"""

