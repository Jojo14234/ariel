from typing import Any

import torch
import numpy as np
import mujoco as mj

from a2_exp.lib import EAPolicy

def _load_genome_to_network(genome: np.ndarray, network: torch.nn.Module) -> None:
    n_parameters = sum(p.numel() for p in network.parameters())
    assert genome.shape == (n_parameters,)

    with torch.no_grad():
        i = 0
        for p in network.parameters():
            j = i + p.numel()
            p.copy_(torch.tensor(genome[i: j], dtype=p.dtype).view_as(p))
            i = j


class NNPolicy(EAPolicy):

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def bind(self, genome: Any):
        _load_genome_to_network(genome, self.network)
        return self

    def __init__(self, out_features: int = 8):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=29, out_features=5, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=5, out_features=out_features, dtype=torch.float64),
            torch.nn.Tanh(),
        )

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        x = np.concatenate([mj_data.qpos, mj_data.qvel], axis=0)
        return self.network(torch.from_numpy(x)).detach().numpy() * np.pi / 2
