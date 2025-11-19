#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # prepare_inputs_labels_for_multimodal() 把文本里的 <image> 占位符替换为视觉特征序列，
    # 统一处理 padding/attention mask，使 scene tokens 与文本 token 共用自注意力空间 
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, image_features, image_sizes
    ):
        if  image_features is None or input_ids.shape[1] == 1:
            # 如果没有图像特征，则不需要多模态融合，直接返回原始输入
            # # 注释掉的代码段处理了past_key_values不为None的情况，可能是用于增量解码
            # if past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
            #     target_shape = past_key_values[-1][-1].shape[-2] + 1
            #     attention_mask = torch.cat((attention_mask, torch.ones(
            #         (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
            #         dtype=attention_mask.dtype,
            #         device=attention_mask.device
            #     )), dim=1)
            #     position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        # 处理图像特征格式： 将列表格式的图像特征转换为连续的特征序列，允许每个批次中包含多张图像
        # 支持列表格式(多视角场景) 和 张量格式 输入
        # 对于列表格式: 双循环将不同视角、不同样本的图像特征重新排序，确保同一样本的不同视角特征连续排列
        # 对于张量格式: 直接将图像特征重塑为 (B, 513, 4096) 的形状，其中 B 是批次大小，513 是图像特征的数量，4096 是每个特征的维度
        # 513 = 14*14*3 + 1
        # 与 torch.flatten()操作的核心目的相同，将多维张量沿着特定维度转换为较低维度的张量

        """
        输入列表格式示例: 包含2个视角的图像特征，每个视角包含3张图像，每个图像的特征形状为
        特征形状: （3, 14, 14, 4096）    (批次大小，高度，宽度，特征维度)
        torch_features_list = [torch.randn(3, 14, 14, 4096), torch.randn(3, 14, 14, 4096)]
          

        """
        if isinstance(image_features,list):
            temp_image_features = []
            for b_id in range(len(image_features[0])):          # 遍历批次中的每个样本
                for img_id in range(len(image_features)):       # 遍历每个视角
                    temp_image_features.append(image_features[img_id][b_id])
            image_features = temp_image_features
        else:
            # 将张量格式的图像特征重塑为 (B, 513, 4096) 的形状，其中 B 是批次大小，513 是图像特征的数量，4096 是每个特征的维度
            # -1维度的大小 = 总元素数量 / (其他所有维度大小的乘积)
            """
            # 创建一个多维张量
            tensor = torch.randn(2, 3, 4, 5)
            print("原始形状:", tensor.shape)  # 输出: 原始形状: torch.Size([2, 3, 4, 5])

            # 展平为二维张量，第一维保持不变
            flattened = tensor.reshape(2, -1)
            print("展平后形状:", flattened.shape)  # 输出: 展平后形状: torch.Size([2, 60])

            # 完全展平为一维张量
            fully_flattened = tensor.reshape(-1)
            print("完全展平形状:", fully_flattened.shape)  # 输出: 完全展平形状: torch.Size([120])
            
            """
            image_features = image_features.reshape(image_features.shape[0], -1, self.hidden_size).to(dtype=self.dtype) # (B, 513, 4096)

        """
        上述列表格式输入示例输出: image_features 是一个包含 3*2 = 6 个元素的列表，每个元素形状为 (14, 14, 4096)
        列表元素的顺序为：   第一个视角的第一张图像特征，第一个视角的第二张图像特征，第一个视角的第三张图像特征，
                            第二个视角的第一张图像特征，第二个视角的第二张图像特征，第二个视角的第三张图像特征

        物理意义: 将多视角的图像特征展平为一个连续的特征序列，每个元素对应一个图像的特征向量，并确保同一样本的不同视角特征连续排列
        
        
        
        张量格式输入示例: image_features = torch.randn(3, 14, 14, 4096)
        输出:  
        """

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
        if attention_mask is None:          # 注意力掩码
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)           # 创建全为1的形状相同张量
        else:
            attention_mask = attention_mask.bool() # (B, 76)            # .bool() 将数据转换为布尔类型
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device) # (76,)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)   # 创建一个与input_ids形状相同的张量，所有元素初始化为IGNORE_INDEX
            # IGNORE_INDEX = -100：在PyTorch中，这是一个特殊值，表示在计算损失函数时应该忽略这些位置


        
        # remove the padding using attention_mask -- TODO: double check
        # 根据掩码保留必要数据id 和 标签
        # 列表推导式， 同时遍历 input_ids 和 attention_mask 两个列表中的元素，
        # cur_attention_mask.cpu() 将掩码张量转移至 CPU 上    TODO 为什么转移？ 面试题？
        # cur_input_ids 只保留了掩码为 True 的位置对应的输入 ID，
        input_ids = [cur_input_ids[cur_attention_mask.cpu()] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_input_ids = []
        cur_image_idx = 0       # 当前处理的图像索引
        
        #TODO 小米好像也是复制过来的,不知道原理
        # 将文本序列中 <image> 占位符(IMAGE_TOKEN_INDEX)替换为实际的图像特征序列, 同时处理对应标签和输入ID
        for batch_idx, cur_input_ids in enumerate(input_ids): # 遍历batch samples
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum() # 只有一个-200，位置在index 35，估计对应的是句子中image的占位   # 统计当前样本中图像占位符的数量
            if num_images == 0:    # 没有显式图像占位符,
                cur_image_features = image_features[cur_image_idx]      # 对应image的图像特征
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)       # 通过 embed_tokens 将文本 token 转换为对应的嵌入向量
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # 最终生成的图像标记位置列表， 格式为[-1, 图像位置1, 图像位置2, ..., 序列长度]
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]] # [-1, 35, 76]
            cur_input_ids_noim = []    # 存储去除图像标记后的文本片段ID列表
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
                    # torch.full(size, fill_value, dtype=None, device=None, requires_grad=False)
                    # size: 新张量的形状(维度), fill_value: 填充值, dtype: 数据类型, device: 所在设备, requires_grad: 是否需要计算梯度
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
