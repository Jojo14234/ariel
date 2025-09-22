from concurrent.futures.process import ProcessPoolExecutor as PPool
from datetime import datetime
from typing import Callable, TypedDict
import numpy as np
from matplotlib import pyplot as plt

from a2_exp.lib import Experiment, EAStrategy
from a2_exp.policies.random_policy import RandomPolicy
from a2_exp.policies.sine_policy import SinePolicy
from a2_exp.strategies import CMAES, GA, RandomStrat
from a2_exp.utils import DummyPool


# print("imports finished!")

class ExperimentRun(TypedDict):
    es_name: str
    seed: int
    population_size: int
    frequency_opt: int


class EAExperiment(Experiment):
    def __init__(self, n_generations: int = 100):
        super().__init__()
        self.n_generations = n_generations

    def run_random(self):
        factory = RandomPolicy
        scores = []
        for seed in 42, 43, 44:
            es = RandomStrat(seed=seed, n_parameters=1, population_size=100)
            scores.append(self.run_single(es, factory)[0])
        self.save("random", scores)

    def run_all(self):
        configs = [
            dict(es_name=es_name, seed=seed, population_size=population_size, frequency_opt=frequency_opt)
            for population_size in (100, 50, 20)
            for frequency_opt in (0, 1, 2)
            for seed in (42, 43, 44)
            for es_name in ("CMAES", "GA")
        ]
        es_by_name = {"CMAES": CMAES, "GA": GA}
        _start_dt = datetime.now()
        print(f"n_configs: {len(configs)}")

        for i, config in enumerate(configs, start=1):
            name = ','.join(f"{k}_{v}" for k, v in config.items())
            if self.exists(name):
                continue

            now = datetime.now()
            estim = now + (now - _start_dt) * (len(configs) - i) / i
            fmt = "%Y-%m-%d %H:%M:%S"
            print(f"{now.strftime(fmt)} | estim: {estim.strftime(fmt)} | {i} / {len(configs)} | {name}")
            factory = lambda: SinePolicy(frequency_opt=config["frequency_opt"])
            es = es_by_name[config["es_name"]](
                n_parameters=factory().n_parameters(),
                seed=config["seed"],
                population_size=config["population_size"]
            )
            scores, genome = self.run_single(es, factory)
            self.save(
                name=name,
                obj={'config': config, 'scores': scores, 'policy': factory().bind(genome)},
            )

    def run(self):
        # from a2_exp.policies.nn_policy import NNPolicy
        # factory = lambda: NNPolicy()
        factory = lambda: SinePolicy(out_features=self.mj_model.nu)
        es_kwargs = {
            "n_parameters": factory().n_parameters(),
            "population_size": 100,
            "seed": 42,
        }

        for i in range(10):
            seed = 42 + i
            score, _ = self.run_single(CMAES(**es_kwargs | {'seed': seed}), factory)
            self.save(f"CMA_seed_{seed}_fixed_reverse", score)
            plt.plot(score, label=f'seed {seed}')
        plt.show()

    def run_single(self, es: EAStrategy, policy_factory: Callable, use_mp: bool = True):
        best_genomes = []
        best_scores = []

        with (DummyPool, PPool)[use_mp]() as pool:
            for i_generation in range(self.n_generations):
                genomes = es.ask()
                policies = [policy_factory().bind(g) for g in genomes]
                futures = [pool.submit(self.evaluate, self.mj_model, p, self.fitness) for p in policies]
                scores = [fut.result() for fut in futures]
                es.tell(genomes, [-f for f in scores])
                argmax = max(range(len(scores)), key=scores.__getitem__)
                print(f"generation {i_generation}, max fitness: {scores[argmax]:.4f}")
                best_genomes.append(genomes[argmax])
                best_scores.append(scores[argmax])
                if es.stop():
                    print(f'early stopping at generation {i_generation} ...')
                    break

        argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        return best_scores, best_genomes[argmax]

    def view_weights(self, name: str):
        policy = self.load(name)['policy']
        print(policy.b)
        print(policy.m)

    def final_plot(self):
        scores = dict(
            cma=[
                self.load(f'es_name_CMAES,seed_{seed},population_size_100,frequency_opt_0')['scores']
                for seed in (42, 43, 44)
            ],
            ga=[
                self.load(f'es_name_GA,seed_{seed},population_size_100,frequency_opt_0')['scores']
                for seed in (42, 43, 44)
            ],
            baseline=self.load('random')
        )
        for name, score in scores.items():
            y = np.array(score)
            mu = np.mean(y, axis=0)
            std = np.std(y, axis=0)
            x = list(range(len(mu)))
            plt.plot(x, mu, label=name)
            plt.fill_between(x, mu + std, mu - std, alpha=0.2)

        plt.show()


def main():
    # EAExperiment().run_all()
    # EAExperiment().run_random()
    EAExperiment().final_plot()
    # EAExperiment().view_weights("es_name_CMAES,seed_42,population_size_100,frequency_opt_0")


if __name__ == '__main__':
    main()
