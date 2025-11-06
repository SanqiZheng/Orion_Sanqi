# 规划模块在启用生成 token 时实例化 present/future 两个 DistributionModule（卷积 encoder + 1D 聚合）和 
# PredictModel GRU，latent 维度 32、隐藏通道 4096/4，用于学习语义推理到轨迹的生成映射 
class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
        self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        # self.encoder = DistributionEncoder2D(
        #     in_channels,
        #     self.compress_dim,
        # )

        self.encoder = DistributionEncoder1DV2(
            in_channels,
            self.compress_dim,
        )

        self.last_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
        )

    def forward(self, s_t):
        encoding = self.encoder(s_t.permute(0, 2, 1).float())
        mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma
    


# distribution_forward() 先基于规划 token 得到 present 分布 (μ, σ)，若提供 GT 未来轨迹则拼接后得到 
# future 分布；训练用 future 分布采样、推理用 present 分布，采样结果再沿时间步展开成 GRU 输入
def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):

    b = present_features.shape[0]
    c = present_features.shape[1]
    present_mu, present_log_sigma = self.present_distribution(present_features)

    future_mu, future_log_sigma = None, None
    if future_distribution_inputs is not None:
        future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
        future_mu, future_log_sigma = self.future_distribution(future_features)

    if noise is None:
        if self.training:
            noise = torch.randn_like(present_mu)
        else:
            noise = torch.randn_like(present_mu)
    if self.training:
        mu = future_mu
        sigma = torch.exp(future_log_sigma)
    else:
        mu = present_mu
        sigma = torch.exp(present_log_sigma)
    sample = mu + sigma * noise

    sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)

    output_distribution = {
        'present_mu': present_mu,
        'present_log_sigma': present_log_sigma,
        'future_mu': future_mu,
        'future_log_sigma': future_log_sigma,
    }
    return sample, output_distribution





# 把 latent sample 复制到 6 个未来时间步，配合由规划 token 切成 4 层的隐藏状态初始化 GRU；
# 输出的未来状态与当前状态拼接形成 states_hs，随后逐时间步送入 ego_fut_decoder 生成 6 模式 × 6 时间步的 2D 轨迹点
def future_states_predict(self, batch_size, sample, hidden_states, current_states):

    future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
    future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)

    hidden_states = hidden_states.permute(1,0,2) # (4, 1, 4096) -> (1, 4, 4096)
    hidden_state = hidden_states.reshape(self.layer_dim, -1, int(4096/4)) # (4, 4, 1024)
    future_states = self.predict_model(future_prediction_input, hidden_state)

    current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
    future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])

    if self.with_cur:
        states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
    else:
        states_hs = future_states_hs

    return states_hs, future_states_hs



# 通过 KL 散度把 present/future 分布拉近，从而约束“语言推理空间”(planning token) 与 “行动空间”(轨迹) 的一致性；
# 采样→GRU→解码流程完成 reasoning space → latent space → action space 的连续映射
@LOSSES.register_module()
class ProbabilisticLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, output, valid_mask):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )
        kl_div = kl_div * valid_mask.any(dim=-1).unsqueeze(-1).unsqueeze(-1)
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1)) * self.loss_weight

        return kl_loss

