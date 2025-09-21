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

    def __init__(self, in_features: int = 5, out_features: int = 8):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, dtype=torch.float64),
        )
        self._frequencies = torch.from_numpy(.1 ** np.arange(-2, in_features-2)).type(torch.float64)

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        inp = self._frequencies.mul(mj_data.time).sin()
        return self.network(inp).detach().numpy()
