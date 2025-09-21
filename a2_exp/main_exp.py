from concurrent.futures.process import ProcessPoolExecutor as PPool
from typing import Callable

from matplotlib import pyplot as plt

from a2_exp.lib import Experiment, EAStrategy
from a2_exp.policies.sine_policy import SinePolicy
from a2_exp.strategies import CMAES
from a2_exp.utils import DummyPool

print("imports finished!")


class EAExperiment(Experiment):
    def __init__(self, n_generations: int = 100):
        super().__init__()
        self.n_generations = n_generations

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
            score, _ = self.run_single(CMAES(**es_kwargs | {'seed':seed }), factory)
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

def main():
    EAExperiment().run()


if __name__ == '__main__':
    main()
