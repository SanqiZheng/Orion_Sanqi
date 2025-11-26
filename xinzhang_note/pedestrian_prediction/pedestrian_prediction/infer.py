import argparse
from pathlib import Path

import numpy as np
import torch

from ped_pred.config import TrajPredConfig
from ped_pred.models import TrajectoryPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for pedestrian trajectory predictor")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--input_npz", type=str, required=True, help="Input npz with 'past' field")
    parser.add_argument(
        "--output_npz",
        type=str,
        required=True,
        help="Output npz path to save predictions (pred_trajs, pred_probs)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt = torch.load(args.model_ckpt, map_location=device)
    cfg_dict = ckpt.get("config", {})
    cfg = TrajPredConfig(**cfg_dict)

    model = TrajectoryPredictor(
        t_obs=cfg.t_obs,
        t_pred=cfg.t_pred,
        num_modes=cfg.num_modes,
        input_dim=2,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_gru_layers,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # load input
    data = np.load(args.input_npz)
    past = data["past"].astype(np.float32)  # [N, T_obs, 2]
    if past.ndim != 3 or past.shape[-1] != 2:
        raise ValueError(f"past must be [N, T_obs, 2], got {past.shape}")

    with torch.no_grad():
        past_t = torch.from_numpy(past).to(device)
        pred_trajs_t, pred_probs_t = model(past_t)

    pred_trajs = pred_trajs_t.cpu().numpy()  # [N, K, T_pred, 2]
    pred_probs = pred_probs_t.cpu().numpy()  # [N, K]

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        pred_trajs=pred_trajs,
        pred_probs=pred_probs,
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

