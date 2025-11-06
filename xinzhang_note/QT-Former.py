def post_update_memory(self, img_metas, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict, rec_can_bus, history_query=None):
    if self.training and mask_dict and mask_dict['pad_size'] > 0:
        rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
        rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
        out_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
        rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
    else:
        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        out_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
        rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
    
    # topk proposals
    # 维护历史 query 的 embedding/reference/timestamp/ego pose，并在每帧把新的 top-k proposals 写入 memory，形成 query-based memory bank 来承载长期历史
    _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
    rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
    rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
    rec_memory = topk_gather(out_memory, topk_indexes).detach()
    rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
    rec_velo = topk_gather(rec_velo, topk_indexes).detach()
    self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
    self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
    if self.use_memory:
        self.scene_memory_timestamp = torch.cat([torch.zeros_like(self.scene_memory_timestamp[:, :self.num_memory,:], dtype=torch.float64), self.scene_memory_timestamp], dim=1)
        self.scene_memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
    self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
    self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
    self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
    self.memory_canbus = torch.cat([rec_can_bus, self.memory_canbus], dim=1)
    self.his_memory_canbus_len += 1 
    self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
    self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
    self.sample_time -= data['timestamp']
    self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
    for i, his_len in enumerate(self.his_memory_canbus_len):
        self.memory_canbus[i:i+1, :his_len.to(torch.int64), 1:4] += data['can_bus'][i:i+1, :3].unsqueeze(1)
        self.memory_canbus[i:i+1, :his_len.to(torch.int64), -1] += data['can_bus'][i:i+1, -1].unsqueeze(1)
    self.memory_scene_tokens = [meta['scene_token'] for meta in img_metas]
    if self.use_memory:
        self.memory_scene_query = torch.cat([history_query.detach(), self.memory_scene_query], dim=1)
    return out_memory

# 将历史 reference point 归一化后生成 positional encoding， 用 ego pose / 时间 编码对齐。并按 num_propagated 把一部分
# 历史query 拼接到当前 query 上，实现 propagation
# 注意：这里的 reference point 是归一化后的，所以要先把 ego pose 对齐，再把时间对齐
def temporal_alignment(self, query_pos, tgt, reference_points):
    B = query_pos.size(0)

    temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
    temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, self.n_control))) 
    temp_memory = self.memory_embedding
    rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
    
    if self.with_ego_pos:
        rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
        rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
        memory_ego_motion = torch.cat([self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
        memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
        temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)

    query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
    temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

    if self.num_propagated > 0:
        tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
        reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
        temp_memory = temp_memory[:, self.num_propagated:]
        temp_pos = temp_pos[:, self.num_propagated:]
        
    return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose





# 解码器层内把当前 query 与拼接后的 temp_memory 做 self-attention，因此历史/当前查询在 transformer 内对话，
# 随后再与多视角特征 cross-attention，达到“query-based 压缩 + 多视角融合” 
def forward(self, query, key, query_pos, key_pos, attn_mask, temp_memory=None, temp_pos=None):
    """ Forward function for transformer decoder layer
    Args:
        query: shape [num_query, batch_size, embed_dims]
        key: shape [num_key, batch_size, embed_dims]
        value: shape [num_value, batch_size, embed_dims]
        query_pos: shape [num_query, batch_size, embed_dims]
        key_pos: shape [num_key, batch_size, embed_dims]
        attn_mask: shape [batch_size, num_query, num_key]
    """
    # TODO: maybe we shouldn't use hard-code layer here
    # TODO: add temporal query here
    # 1. Multi-head Self-attention (between queries)
    if temp_memory is not None:
        temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
        temp_pos = torch.cat([query_pos, temp_pos], dim=1)
    else:
        temp_key = temp_value = query
        temp_pos = query_pos

    query, attn0 = self.transformer_layers[0](query, temp_key, temp_value, query_pos, temp_pos, attn_mask=attn_mask)
    # 2. LayerNorm
    query = self.transformer_layers[1](query)
    # 3. Multi-head Cross-attention (between queries and keys)
    query, attn1 = self.transformer_layers[2](query, key, key, query_pos, key_pos, attn_mask=None)
    # 4. LayerNorm
    query = self.transformer_layers[3](query)
    # 5. Feed-forward Network
    query = self.transformer_layers[4](query)
    # 6. LayerNorm
    query = self.transformer_layers[5](query)

    return query