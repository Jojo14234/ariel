import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import mujoco as mj

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import load_graph_from_json
from ariel.simulation.environments import OlympicArena

TARGET_POSITION = [5, 0, 0.5]
CDIR = Path(__file__).parent
GRAPH_JSON = CDIR / "best_graph.json"


def _load_genome_to_network(genome: np.ndarray, network: torch.nn.Module) -> None:
    n_parameters = sum(p.numel() for p in network.parameters())
    assert genome.shape == (n_parameters,)

    with torch.no_grad():
        i = 0
        for p in network.parameters():
            j = i + p.numel()
            p.copy_(torch.tensor(genome[i: j], dtype=p.dtype).view_as(p))
            i = j


class NNPolicy:
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

    def bind(self, genome: np.ndarray):
        _load_genome_to_network(genome.copy(), self.network)
        return self


def fitness(_: mj.MjModel, mj_data: mj.MjData):
    distance = np.sqrt(sum((b - a) ** 2 for a, b in zip(mj_data.geom('robot1_core').xpos, TARGET_POSITION)))
    return -distance


def simulate(sim_time: int = 20, n_steps_per_cycle: int = 10):
    robot_graph = load_graph_from_json(GRAPH_JSON)
    robot_core = construct_mjspec_from_graph(robot_graph).spec
    spec = (w := OlympicArena()).spawn(robot_core) or w.spec
    mj_model = spec.compile()
    mj_data = mj.MjData(mj_model)
    mj.mj_resetData(mj_model, mj_data)

    while mj_data.time < sim_time:
        mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
        mj_data.ctrl = np.clip(CONTROLLER(mj_model, mj_data), -np.pi / 2, np.pi / 2)

    print(f"fitness: {fitness(mj_model, mj_data):.3f}")


with open(CDIR / "best_brain.pkl", "rb") as f:
    brain_genome = pickle.load(f)

CONTROLLER = NNPolicy(37, 12).bind(brain_genome)

if __name__ == '__main__':
    simulate(sim_time=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
