import numpy as np

from a3_exp.lib import EAStrategy


class CMAES(EAStrategy):
    """
    CMA-ES using the CMA package
    """

    def __init__(self, n_parameters: int, population_size: int, seed: int):
        super().__init__(n_parameters=n_parameters, population_size=population_size)

        import cma  # import is long

        options = {
            "seed": seed,
            "popsize": population_size,
            'verb_disp': 0,
        }
        self.cma = cma.CMAEvolutionStrategy([1.] * n_parameters, .5, options)

    def ask(self):
        return self.cma.ask()

    def tell(self, samples, scores):
        return self.cma.tell(samples, scores)

    def stop(self):
        # return self.cma.stop()
        return self and False

class RandStrat(EAStrategy):
    def __init__(self, n_parameters: int, population_size: int, seed: int):
        super().__init__(n_parameters=n_parameters, population_size=population_size)
        self.rng = np.random.default_rng(seed)
        self._population = [self.rng.random(size=self.n_parameters) for _ in range(self.population_size)]


    def ask(self):
        return self._population

    def tell(self, samples, scores):
        keep = int(self.population_size * .5)
        best_samples = map(samples.__getitem__, sorted(range(len(samples)), key=scores.__getitem__)[:keep])
        self._population = list(best_samples) + [
            self.rng.random(size=self.n_parameters) for _ in range(self.population_size - keep)
        ]
        assert len(self._population) == self.population_size

    def stop(self):
        return self and False
