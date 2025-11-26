"""

def film_block(features, conditioning_vector):
    # 从条件向量中生成 scale / shift 参数
    scale = mlp_scale(conditioning_vector)              # 每个通道的缩放系数
    shift = mlp_shift(conditioning_vector)              # 每个通道的偏移值

    # 每个通道分别调整，保持特征形状一致
    return features * scale[:, None, None] + 
"""