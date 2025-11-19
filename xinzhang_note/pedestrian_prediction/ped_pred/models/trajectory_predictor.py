from typing import Tuple

import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    """Lightweight multi-modal pedestrian trajectory predictor.

    Input:
      - past: [B, T_obs, 2]
    Output:
      - pred_trajs: [B, K, T_pred, 2]
      - pred_probs: [B, K]
    """

    def __init__(
        self,
        t_obs: int,
        t_pred: int,
        num_modes: int = 3,
        input_dim: int = 2,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.num_modes = num_modes

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 每个 mode 对应一个独立的 decoder MLP，输出整个未来轨迹（相对位移）
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, t_pred * 2),
                )
                for _ in range(num_modes)
            ]
        )

        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modes),
        )

    def forward(self, past: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            past: [B, T_obs, 2]
        Returns:
            pred_trajs: [B, K, T_pred, 2]
            pred_probs: [B, K] (softmax over K)
        """
        B, T_obs, D = past.shape
        assert T_obs == self.t_obs, f"T_obs mismatch: expected {self.t_obs}, got {T_obs}"

        x = self.input_proj(past)  # [B, T_obs, embed_dim]
        _, h_n = self.encoder(x)   # h_n: [num_layers, B, hidden_dim]
        h = h_n[-1]                # [B, hidden_dim]

        preds = []
        for dec in self.decoders:
            out = dec(h)  # [B, T_pred * 2]
            out = out.view(B, self.t_pred, 2)
            # 预测相对位移，再累加到最后一个观测点上，得到平滑的未来轨迹
            rel = out
            last_pos = past[:, -1:, :]  # [B, 1, 2]
            full = last_pos + torch.cumsum(rel, dim=1)
            preds.append(full)

        pred_trajs = torch.stack(preds, dim=1)  # [B, K, T_pred, 2]
        logits = self.prob_head(h)              # [B, K]
        pred_probs = torch.softmax(logits, dim=-1)

        return pred_trajs, pred_probs

    @staticmethod
    def best_of_k_loss(
        pred_trajs: torch.Tensor,
        gt_future: torch.Tensor,
        pred_probs: torch.Tensor,
        mode: str = "min_ade",
    ) -> torch.Tensor:
        """Best-of-K regression loss with optional probability regularization.

        Args:
            pred_trajs: [B, K, T_pred, 2]
            gt_future:  [B, T_pred, 2]
            pred_probs: [B, K]
            mode: "min_ade" or "min_fde"
        """
        B, K, T_pred, _ = pred_trajs.shape

        gt = gt_future.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, T_pred, 2]
        l2 = torch.norm(pred_trajs - gt, dim=-1)           # [B, K, T_pred]

        if mode == "min_fde":
          # final displacement error
          errors = l2[:, :, -1]  # [B, K]
        else:
          # average displacement error
          errors = l2.mean(dim=-1)  # [B, K]

        min_errors, min_indices = errors.min(dim=1)  # [B]

        # 回归误差
        regression_loss = min_errors.mean()

        # 可选：鼓励选中的 mode 概率更高（负对数似然）
        chosen_probs = pred_probs[torch.arange(B, device=pred_probs.device), min_indices]
        nll = -torch.log(chosen_probs.clamp(min=1e-6)).mean()

        return regression_loss + 0.1 * nll

