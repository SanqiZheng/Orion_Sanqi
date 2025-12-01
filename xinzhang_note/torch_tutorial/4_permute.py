tensor.permute(dim0, dim1, dim2, ...)
# - 参数 ：需要重新排列的维度顺序，用整数表示
# - 返回值 ：维度重新排列后的新张量
# - 特点 ：返回的是原张量的视图（view），不会复制数据，内存共享
# permute 是PyTorch中用于 重新排列张量维度 的重要函数，它可以帮助我们灵活地调整张量的形状以适应不同的神经网络层需求。


"""
### . 与其他维度调整函数的区别
函数 功能 特点 permute 任意重排维度 可同时调整多个维度 transpose 交换两个维度 一次只能交换两个维度 view/reshape 改变形状 需要保证元素总数不变 unsqueeze/squeeze 增加/减少维度 仅影响维度数量

"""

encoding = self.encoder(s_t.permute(0, 2, 1).float())       # 特征编码
# permute(0, 2, 1) 将维度从 [B, C, T] 调整为 [B, T, C]，便于后续按时间维度拆分均值与对数标准差
mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1)
mu = mu_log_sigma[:, :, :self.latent_dim]               # 高斯分布的均值，形状为 [B, T, latent_dim]
log_sigma = mu_log_sigma[:, :, self.latent_dim:]        # 对数标准差，用于描述高斯分布的方差