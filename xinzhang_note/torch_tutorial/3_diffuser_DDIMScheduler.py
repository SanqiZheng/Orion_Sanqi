from diffusers.schedulers import DDIMScheduler
self.diffusion_scheduler = DDIMScheduler(
    num_train_timesteps=1000,        # 训练时间步数
    beta_schedule="scaled_linear",  # β调度方式
    prediction_type="sample",      # 预测类型
)