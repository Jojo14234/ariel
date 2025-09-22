import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

import mujoco as mj
import numpy as np

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


class Experiment(ABC):
    mj_model: mj.MjModel

    def __init__(self):
        self.mj_model = self.get_default_model()
        self.fitness = self.experiment_fitness

    @staticmethod
    def get_default_model():
        from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
        from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
        spec = (w := SimpleFlatWorld()).spawn(gecko().spec, spawn_position=[0, 0, .1]) or w.spec
        return cast(mj.MjModel, spec.compile())

    @staticmethod
    def experiment_fitness(_: mj.MjModel, mj_data: mj.MjData):
        """
        - baseline fitness is the total position moved in the positive y direction
        - y chosen over x because the robot is aligned in y direction initially
        """
        x, y, z = mj_data.geom('robot-core').xpos
        # return y - abs(x)
        # return abs(y)
        return y

    @staticmethod
    def evaluate(mj_model, policy, fitness, n_cycles: int = 1000, n_steps_per_cycle: int = 10):
        mj_data = mj.MjData(mj_model)

        for _ in range(n_cycles):
            mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
            mj_data.ctrl = policy(mj_model, mj_data)
            mj_data.ctrl = np.clip(mj_data.ctrl, -np.pi / 2, np.pi / 2)

        return fitness(mj_model, mj_data)

    @staticmethod
    def view(mj_model, policy, fitness, n_cycles: int = 1000, n_steps_per_cycle: int = 10):
        mj_data = mj.MjData(mj_model)

        import mujoco.viewer as mjv
        with mjv.launch_passive(mj_model, mj_data) as viewer:
            for _ in range(n_cycles):
                mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
                mj_data.ctrl = policy(mj_model, mj_data)
                mj_data.ctrl = np.clip(mj_data.ctrl, -np.pi / 2, np.pi / 2)
                viewer.sync()
                time.sleep(1 / 60)
        print(f"final fitness: {fitness(mj_model, mj_data)}")

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

    @abstractmethod
    def run(self): ...
