import pickle
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph

TARGET_POSITION = [5, 0, 0.5]
CDIR = Path(__file__).parent


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


def _string_to_graph(graph: str):
    import json
    from networkx.readwrite import json_graph

    return json_graph.node_link_graph(json.loads(graph), edges="edges")


def spec_to_olympic_world(spec: mj.MjSpec, position=(-0.8, 0, 0.1)) -> mj.MjSpec:
    from ariel.simulation.environments import OlympicArena
    spec: mj.MjSpec = (w := OlympicArena()).spawn(spec, position=position) or w.spec

    return spec


def fitness(_: mj.MjModel, mj_data: mj.MjData):
    distance = np.sqrt(sum((b - a) ** 2 for a, b in zip(mj_data.geom('robot1_core').xpos, TARGET_POSITION)))
    return -distance


def simulate(mj_model: mj.MjModel, policy, sim_time: int = 20, n_steps_per_cycle: int = 10):
    mj_data = mj.MjData(mj_model)
    mj.mj_resetData(mj_model, mj_data)
    max_fitness = float("-inf")

    while mj_data.time < sim_time:
        mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
        mj_data.ctrl = np.clip(policy(mj_model, mj_data), -np.pi / 2, np.pi / 2)
        x = fitness(mj_model, mj_data)
        max_fitness = max(max_fitness, x)

    return max_fitness, fitness(mj_model, mj_data)


def main(sim_time: int = 32):
    robot_graph = _string_to_graph((CDIR / "best_graph.json").read_text())
    robot_core = construct_mjspec_from_graph(robot_graph).spec
    spec = spec_to_olympic_world(robot_core)
    mj_model = spec.compile()
    max_fitness, final_fitness = simulate(mj_model, CONTROLLER, sim_time=sim_time)
    print(f"max fitness: {max_fitness:.3f}, final fitness: {final_fitness:.3f}")


with open(CDIR / "best_brain.pkl", "rb") as f:
    brain_genome = pickle.load(f)

CONTROLLER = NNPolicy(37, 12).bind(brain_genome)
GRAPH = _string_to_graph((CDIR / "best_graph.json").read_text()) # DO NOT LOAD WITH 'load_graph_from_json'
SIM_DURATION = 28

if __name__ == '__main__':
    main(sim_time=SIM_DURATION)
