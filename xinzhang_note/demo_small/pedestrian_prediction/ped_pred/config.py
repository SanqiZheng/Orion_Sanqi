import dataclasses
from dataclasses import dataclass


@dataclass
class TrajPredConfig:
    # data
    t_obs: int = 8
    t_pred: int = 12
    num_modes: int = 3

    # model size
    input_dim: int = 2
    embed_dim: int = 64
    hidden_dim: int = 128
    num_gru_layers: int = 1

    # training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    num_epochs: int = 50
    grad_clip: float = 5.0

    # misc
    device: str = "cuda"

    def to_dict(self):
        return dataclasses.asdict(self)

