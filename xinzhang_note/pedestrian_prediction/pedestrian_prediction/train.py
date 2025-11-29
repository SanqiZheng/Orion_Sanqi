import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ped_pred.config import TrajPredConfig
from ped_pred.data import PedestrianTrajDataset
from ped_pred.models import TrajectoryPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pedestrian trajectory predictor")
    parser.add_argument("--train_npz", type=str, required=True, help="Path to training npz file")
    parser.add_argument("--val_npz", type=str, required=True, help="Path to validation npz file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_modes", type=int, default=3)
    parser.add_argument("--t_obs", type=int, default=8)
    parser.add_argument("--t_pred", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # config
    cfg = TrajPredConfig(
        t_obs=args.t_obs,
        t_pred=args.t_pred,
        num_modes=args.num_modes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
    )

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # data
    train_ds = PedestrianTrajDataset(args.train_npz)
    val_ds = PedestrianTrajDataset(args.val_npz)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # model
    model = TrajectoryPredictor(
        t_obs=cfg.t_obs,
        t_pred=cfg.t_pred,
        num_modes=cfg.num_modes,
        input_dim=2,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_gru_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val_loss = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        for past, future, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [train]"):
            past = past.to(device)     # [B, T_obs, 2]
            future = future.to(device) # [B, T_pred, 2]

            optimizer.zero_grad()
            pred_trajs, pred_probs = model(past)
            loss = TrajectoryPredictor.best_of_k_loss(pred_trajs, future, pred_probs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1

        train_loss = train_loss_sum / max(num_batches, 1)

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for past, future, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [val]"):
                past = past.to(device)
                future = future.to(device)
                pred_trajs, pred_probs = model(past)
                loss = TrajectoryPredictor.best_of_k_loss(pred_trajs, future, pred_probs)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()

