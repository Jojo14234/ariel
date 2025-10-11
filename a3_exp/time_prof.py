import pickle
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
import torch
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena

torch.manual_seed(42) # SHOULD MAKE NDE DETERMINISTIC

TARGET_POSITION = [5, 0, 0.5]
NUM_OF_MODULES = 30
NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
HPD = HighProbabilityDecoder(NUM_OF_MODULES)

_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
if not _ARTIFACT_DIR.exists():
    _ARTIFACT_DIR.mkdir()
assert _ARTIFACT_DIR.is_dir()


def simulate(mj_model: mj.MjModel, policy, fitness, sim_time: int = 20, n_steps_per_cycle: int = 10):
    mj_data = mj.MjData(mj_model)
    while mj_data.time < sim_time:
        mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
        mj_data.ctrl = np.clip(policy(mj_model, mj_data), -np.pi / 2, np.pi / 2)

    return fitness(mj_model, mj_data)


def default_fitness(_: mj.MjModel, mj_data: mj.MjData):
    distance = np.sqrt(sum((b - a) ** 2 for a, b in zip(mj_data.geom('robot1-core').xpos, TARGET_POSITION)))
    return -distance


def _genotype_to_graph(genotype: Any) -> DiGraph:
    return HPD.probability_matrices_to_graph(*NDE.forward(list(genotype)))


def _graph_to_spec(graph: DiGraph) -> mj.MjSpec:
    return construct_mjspec_from_graph(graph).spec


def _graph_to_string(graph: DiGraph) -> str:
    import json
    from networkx.readwrite import json_graph
    return json.dumps(json_graph.node_link_data(graph, edges="edges"), indent=4)


def _string_to_graph(graph: str) -> DiGraph:
    import json
    from networkx.readwrite import json_graph

    return json_graph.node_link_graph(json.loads(graph), edges="edges")

def save(name: str, obj: Any) -> None:
    with open((_ARTIFACT_DIR / name).with_suffix(".pkl"), "wb") as f:
        pickle.dump(obj, f)

def load(name: str):
    with open((_ARTIFACT_DIR / name).with_suffix(".pkl"), "rb") as f:
        return pickle.load(f)

def exists(name: str) -> bool:
    return ((_ARTIFACT_DIR / name).with_suffix(".pkl")).exists()

def deterministic_run():
    seeded_rng = np.random.default_rng(42)
    genotype = seeded_rng.normal(size=(3, 64))
    graph = _genotype_to_graph(genotype)
    robot_spec = _graph_to_spec(graph)

    world = OlympicArena()
    world.spawn(robot_spec)

    mj_model = world.spec.compile()
    policy = lambda _, d: d.ctrl + seeded_rng.normal(size=d.ctrl.shape)
    fitness = simulate(mj_model, policy, default_fitness)

    return {
        'graph_str': _graph_to_string(graph),
        'robot_spec_str': robot_spec.to_xml(),
        'world_str': world.spec.to_xml(),
        'fitness': fitness,
    }

def main(name: str = "exp-1"):
    if not exists(name):
        print("no previous results found, running and saving ...")
        save(name, deterministic_run())
        return

    print("comparing vs previous results ...")
    prev_results = load(name)
    curr_results = deterministic_run()

    for k in curr_results:
        print(f"{k}: prev == curr? {prev_results[k] == curr_results[k]}")

    (_ARTIFACT_DIR / f"{name}_world_a").write_text(prev_results['world_str'])
    (_ARTIFACT_DIR / f"{name}_world_b").write_text(curr_results['world_str'])



if __name__ == '__main__':
    main()
