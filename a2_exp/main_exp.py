from concurrent.futures.process import ProcessPoolExecutor as PPool
from typing import Callable

from a2_exp.lib import Experiment, EAStrategy
from a2_exp.policies.sine_policy import SinePolicy
from a2_exp.strategies import CMAES

print("imports finished!")


class DummyResult:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def result(self):
        return self.f(*self.args, **self.kwargs)

class DummyPool:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    def submit(self, f, *args, **kwargs):
        _ = self # to avoid 'can be static' hint
        return DummyResult(f, *args, **kwargs)

class EAExperiment(Experiment):
    def __init__(self, n_generations: int = 100):
        super().__init__()
        self.n_generations = n_generations

    def run(self):
        sin_factory = lambda: SinePolicy(out_features=self.mj_model.nu)
        es_kwargs = {
            "n_parameters": sin_factory().n_parameters(),
            "population_size": 100,
            "seed": 42,
        }

        self.run_single(CMAES(**es_kwargs), sin_factory)

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
                print(f"generation {i_generation}, max fitness: {scores[argmax]}")
                best_genomes.append(genomes[argmax])
                best_scores.append(scores[argmax])
                if es.stop():
                    print(f'early stopping at generation {i_generation} ...')
                    break

        argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        print(f"total best score: {best_scores[argmax]}")

def main():
    # RandomBaseline().run()
    EAExperiment().run()
    # EAExperiment().view()


if __name__ == '__main__':
    main()
