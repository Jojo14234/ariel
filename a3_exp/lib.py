import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self, Any

import mujoco as mj
import numpy as np

from a3_exp.utils import timeit

SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

type Graph = Any # alias to avoid import

_ARTIFACT_DIR = Path(__file__).parent / "artifacts"
if not _ARTIFACT_DIR.exists():
    _ARTIFACT_DIR.mkdir()
assert _ARTIFACT_DIR.is_dir()


class EAPolicy(ABC):

    @abstractmethod
    def bind(self, genome: Any) -> Self: ...

    @abstractmethod
    def n_parameters(self) -> int: ...

    @abstractmethod
    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData): ...


class EAStrategy(ABC):
    def __init__(self, n_parameters: int, population_size: int):
        self.n_parameters = n_parameters
        self.population_size = population_size

    @abstractmethod
    def ask(self): ...

    @abstractmethod
    def tell(self, samples, scores): ...

    @abstractmethod
    def stop(self) -> bool: ...



class Experiment:
    def __init__(self):
        from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
        from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

        self._nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        self._hpd = lambda: HighProbabilityDecoder(NUM_OF_MODULES)

    @staticmethod
    def fitness(_: mj.MjModel, mj_data: mj.MjData):
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(mj_data.geom('robot-core').xpos, TARGET_POSITION)))
        return -distance

    @staticmethod
    def basic_fitness(_: mj.MjModel, mj_data: mj.MjData):
        return -mj_data.geom('robot-core').xpos[1]

    @staticmethod
    def evaluate(mj_model: mj.MjModel, policy, fitness, sim_time: int = 20, n_steps_per_cycle: int = 10):
        mj_data = mj.MjData(mj_model)
        mj.mj_resetData(mj_model, mj_data)
        t_iter = 0.002 * n_steps_per_cycle
        n_iter = int(sim_time / t_iter) + 1
        for _ in range(n_iter):
            mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
            mj_data.ctrl = np.clip(policy(mj_model, mj_data), -np.pi / 2, np.pi / 2)

        # assert 0 < mj_data.time - sim_time < 2 * t_iter, f"seen time {mj_data.time}, expected: {sim_time}~{t_iter}"
        return fitness(mj_model, mj_data)

    @staticmethod
    def view(mj_model: mj.MjModel, policy, fitness, sim_time: int = 20, n_steps_per_cycle: int = 10):
        mj_data = mj.MjData(mj_model)

        import mujoco.viewer as mjv
        with mjv.launch_passive(mj_model, mj_data) as viewer:
            while mj_data.time < sim_time:
                mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
                mj_data.ctrl = np.clip(policy(mj_model, mj_data), -np.pi / 2, np.pi / 2)
                viewer.sync()
                time.sleep(1 / 20)
        print(f"final fitness: {fitness(mj_model, mj_data)}")

    @staticmethod
    def _graph_to_string(graph: Graph) -> str:
        import json
        from networkx.readwrite import json_graph
        return json.dumps(json_graph.node_link_data(graph, edges="edges"), indent=4)

    @staticmethod
    def _string_to_graph(graph: str) -> Graph:
        import json
        from networkx.readwrite import json_graph

        return json_graph.node_link_graph(json.loads(graph), edges="edges")

    @staticmethod
    def _greph_to_mj_spec(graph: Graph) -> mj.MjSpec:
        from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
        return construct_mjspec_from_graph(graph).spec

    @staticmethod
    def spec_to_simple_world(spec: mj.MjSpec) -> mj.MjModel:
        from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
        spec = (w := SimpleFlatWorld()).spawn(spec, spawn_position=[0, 0, .1]) or w.spec
        return spec.compile()

    @staticmethod
    def spec_to_olympic_world(spec: mj.MjSpec) -> mj.MjModel:
        from ariel.simulation.environments import OlympicArena
        spec = (w := OlympicArena()).spawn(spec, spawn_position=SPAWN_POS) or w.spec
        return spec.compile()

    @staticmethod
    def save(name: str, obj: Any) -> None:
        with open(_ARTIFACT_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(name: str):
        with open(_ARTIFACT_DIR / f"{name}.pkl", "rb") as f:
            return pickle.load(f)

    @staticmethod
    def exists(name: str) -> bool:
        return (_ARTIFACT_DIR / f"{name}.pkl").exists()

    def _genotype_to_graph(self, genotype: Any) -> Graph:
        return self._hpd().probability_matrices_to_graph(*self._nde.forward(list(genotype)))
