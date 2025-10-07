from typing import Any, Self

import mujoco as mj
import numpy as np
import torch

from a3_exp.lib import EAPolicy


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

    @classmethod
    def from_model(cls, mj_model: mj.MjModel):
        d = mj.MjData(mj_model)
        return lambda: cls(in_features=len(d.qpos) + len(d.qvel), out_features=mj_model.nu)

    def __init__(self, in_features: int, out_features: int):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=5, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=5, out_features=out_features, dtype=torch.float64),
            torch.nn.Tanh(),
        )

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        x = np.concatenate([mj_data.qpos, mj_data.qvel], axis=0)
        return self.network(torch.from_numpy(x)).detach().numpy() * np.pi / 2

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def bind(self, genome: np.ndarray):
        _load_genome_to_network(genome, self.network)
        return self


class DoublePolicy(NNPolicy):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features + 4, out_features)
        self.freq = np.zeros(4)
        self.lag = np.zeros(4)

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        f = np.sin(self.lag + self.freq * mj_data.time)
        x = np.concatenate([mj_data.qpos, mj_data.qvel, f], axis=0)
        return self.network(torch.from_numpy(x)).detach().numpy() * np.pi / 2

    def bind(self, genome: Any) -> Self:
        self.lag = genome[:4].clip(-20, 20)
        self.freq = .1 ** genome[4:8].clip(-4, 4)
        return super().bind(genome[8:])

    def n_parameters(self) -> int:
        return super().n_parameters() + 8
