import pickle
import time
from abc import ABC, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from typing import cast, Any, Self, Optional

import mujoco as mj
import numpy as np
import torch
from matplotlib import pyplot as plt

print("imports finished!")


def _get_default_model():
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
    from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
    spec = (w := SimpleFlatWorld()).spawn(gecko().spec, spawn_position=[0, 0, .1]) or w.spec
    return cast(mj.MjModel, spec.compile())


def _load_genome_to_network(genome: np.ndarray, network: torch.nn.Module) -> None:
    n_parameters = sum(p.numel() for p in network.parameters())
    assert genome.shape == (n_parameters,)

    with torch.no_grad():
        i = 0
        for p in network.parameters():
            j = i + p.numel()
            p.copy_(torch.tensor(genome[i: j], dtype=p.dtype).view_as(p))
            i = j


def experiment_fitness(_: mj.MjModel, mj_data: mj.MjData):
    """
    baseline fitness is the total position moved in the positive y direction

    chosen y over x because the robot is aligned in y direction initially

    alternatives:
        - could choose `y - abs(x)` so the robot stays in line
    """
    x, y = mj_data.geom('robot-core').xpos[:2]
    # return y - abs(x)
    # return abs(y)
    return y


class Experiment:
    mj_model: mj.MjModel

    def __init__(self):
        self.mj_model = _get_default_model()

    @staticmethod
    def evaluate(mj_model, policy, fitness, n_cycles: int = 1000, n_steps_per_cycle: int = 10):
        mj_data = mj.MjData(mj_model)

        for _ in range(n_cycles):
            mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
            mj_data.ctrl = policy(mj_model, mj_data)
            mj_data.ctrl = np.clip(mj_data.ctrl, -np.pi / 2, np.pi / 2)

        return fitness(mj_model, mj_data)


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        num_joints = mj_model.nu
        hinge_range = np.pi / 2
        rand_moves = self.rng.uniform(low=-hinge_range,  # -pi/2
                                      high=hinge_range,  # pi/2
                                      size=num_joints)

        delta = 0.05
        return mj_data.ctrl + rand_moves * delta


class EAPolicy(ABC):

    @abstractmethod
    def bind(self, genome: Any) -> Self: ...

    @abstractmethod
    def n_parameters(self) -> int: ...


class NNPolicy(EAPolicy):

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def bind(self, genome: Any):
        _load_genome_to_network(genome, self.network)
        return self

    def __init__(self, in_features: int = 5, out_features: int = 10):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, dtype=torch.float64),
        )
        self._frequencies = torch.from_numpy(.1 ** np.arange(-2, in_features-2)).type(torch.float64)

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        inp = self._frequencies.mul(mj_data.time).sin()
        return self.network(inp).detach().numpy()


class RandomBaseline(Experiment):

    def run(self):
        results = [self.evaluate(self.mj_model, RandomPolicy(seed), experiment_fitness) for seed in range(3)]
        print(results, np.mean(results), np.std(results))

class GA:
    def __init__(self, n_parameters: int, pop_size: int = 20, seed: int = 42):
        self.n_parameters = n_parameters
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.population = [self.rng.normal(size=self.n_parameters) for _ in range(self.pop_size)]
        self.crossover_rate = .2
        self.mutation_rate = .1
        self.mutation_magnitude = .1
        self.sampling_size = 3

    def _crossover(self, a, b):
        if self.rng.random() < self.crossover_rate:
            i = self.rng.integers(1, self.n_parameters - 1)
            mask = np.arange(self.n_parameters) < i
            return (a * mask) + (b * ~mask), (a * ~mask) + (b * mask)
        return a, b

    def _blend(self, a, b):
        if self.rng.random() < self.crossover_rate:
            mask = self.rng.random(size=self.n_parameters)
            mask_ = 1 - mask
            return (a * mask) + (b * mask_), (a * mask_) + (b * mask)
        return a, b


    def _mutate(self, genome):
        if self.rng.random() < self.mutation_rate:
            return genome + self.rng.normal(size=self.n_parameters) * self.mutation_magnitude

        return genome

    def ask(self):
        return self.population

    def tell(self, samples, scores):
        assert len(samples) == len(scores) == self.pop_size
        indices = list(range(len(samples)))
        scoring = dict(zip(indices, scores))
        selected_i = [min(self.rng.choice(indices, self.sampling_size), key=scoring.get) for _ in range(self.pop_size)]
        selected = [samples[i] for i in set(selected_i)]
        print(f'{len(selected_i)} -> {len(set(selected_i))}')
        pairs = [self._blend(*self.rng.choice(selected, 2)) for _ in range(self.pop_size)]
        new_generation = [self._mutate(x) for pair in pairs for x in pair]
        new_generation = self.rng.choice(new_generation, size=self.pop_size)
        # new_generation = [self._mutate(x)
        #     for _ in range(self.pop_size)
        #     for pair in self._crossover(*self.rng.choice(selected, 2))
        #     for x in pair
        # ]
        assert len(new_generation) == self.pop_size, f'new gen size {len(new_generation)} should be {self.pop_size}'
        self.population = new_generation

    def stop(self):
        _ = self
        return False




class EAExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.policy_factory = lambda: NNPolicy(out_features=self.mj_model.nu)
        self.n_generations = 100

    def run_single(self):
        # import cma

        policy = self.policy_factory()
        n_params = policy.n_parameters()


        # es = cma.CMAEvolutionStrategy([0] * n_params, 0.5, {'popsize': 30, 'seed': 42})
        es = GA(n_params, pop_size=120)
        best_genomes = []
        best_scores = []

        with ProcessPoolExecutor() as pool:
            for i_generation in range(self.n_generations):
                genomes = es.ask()
                policies = [self.policy_factory().bind(g) for g in genomes]
                futures = [pool.submit(self.evaluate, self.mj_model, p, experiment_fitness) for p in policies]
                scores = [fut.result() for fut in futures]
                es.tell(genomes, [-f for f in scores])
                argmax = max(range(len(scores)), key=scores.__getitem__)
                print(f"generation {i_generation}, max fitness: {scores[argmax]}")
                best_genomes.append(genomes[argmax])
                best_scores.append(scores[argmax])
                if es.stop():
                    print(f'early stopping at generation {i_generation} ...')
                    break

        argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        print(f"total best score: {best_scores[argmax]}")

        # with open(f"best-{n_params}p-cma.pkl", "wb") as f:
        #     print(f"saving {best_genomes[argmax]}")
        #     pickle.dump(best_genomes[argmax], f)
        # plt.plot(best_scores)
        # plt.show()

    def view(self, genome: Optional[np.ndarray] = None):
        if genome is None:
            n_params = self.policy_factory().n_parameters()
            with open(f"best-{n_params}p-cma.pkl", "rb") as f:
                genome = pickle.load(f)

        policy = self.policy_factory().bind(genome)
        model = self.mj_model
        data = mj.MjData(model)

        import mujoco.viewer as mjv
        with mjv.launch_passive(model, data) as viewer:
            for _ in range(1000):
                mj.mj_step(model, data, nstep=10)
                data.ctrl = policy(model, data)
                data.ctrl = np.clip(data.ctrl, -np.pi / 2, np.pi / 2)
                viewer.sync()
                time.sleep(1 / 60)
        print(f"final fitness: {experiment_fitness(model, data)}")

def main():
    # RandomBaseline().run()
    EAExperiment().run_single()
    # EAExperiment().view()


if __name__ == '__main__':
    main()
