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
        self.cma = cma.CMAEvolutionStrategy([0.] * n_parameters, .5, options)

    def ask(self):
        return self.cma.ask()

    def tell(self, samples, scores):
        return self.cma.tell(samples, [-f for f in scores]) # maximise scores, cma does minimization

    def stop(self):
        # return self.cma.stop()
        return self and False

class RandStrat(EAStrategy):
    def __init__(self, n_parameters: int, population_size: int, seed: int):
        super().__init__(n_parameters=n_parameters, population_size=population_size)
        self.rng = np.random.default_rng(seed)
        self._population = [self.rng.normal(size=self.n_parameters) for _ in range(self.population_size)]

    def ask(self):
        return self._population

    def tell(self, samples, scores):
        keep = int(self.population_size * .5)
        best_samples = map(samples.__getitem__, sorted(range(len(samples)), key=scores.__getitem__)[:keep])
        self._population = list(best_samples) + [
            self.rng.normal(size=self.n_parameters) for _ in range(self.population_size - keep)
        ]
        assert len(self._population) == self.population_size

    def stop(self):
        return self and False


class GA(EAStrategy):
    def __init__(
        self,
        n_parameters: int,
        population_size: int,
        seed: int,
        mutation_rate: float,
        crossover_rate: float,
        tournament_size: int = 5,
    ):
        super().__init__(n_parameters, population_size)

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.tournament_size = tournament_size

        self.rng = np.random.default_rng(seed)
        self.population = [self.rng.random(size=self.n_parameters) for _ in range(self.population_size)]

    def ask(self):
        return self.population

    def tell(self, samples, scores):
        assert len(samples) == len(scores) == self.population_size
        """
        # Tournament
        """
        self.population = samples.copy()
        idxs = sorted(range(len(samples)), key=lambda i: scores[i], reverse=True)[:1]
        best = [self.population[idx].copy() for idx in idxs]

        scores = np.array(scores)
        parent_i = []
        for _ in range(self.population_size):
            idx = self.rng.choice(self.population_size, self.tournament_size, replace=False)
            parent_i.append(idx[np.argmax(scores[idx])])

        parents = [self.population[i] for i in parent_i]

        """
        # Crossover
        """
        offspring = []
        for i in range(0, self.population_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % self.population_size]
            if self.rng.random() < self.crossover_rate:
                cx_point = self.rng.integers(1, self.n_parameters)
                c1 = np.concatenate([p1[:cx_point], p2[cx_point:]])
                c2 = np.concatenate([p2[:cx_point], p1[cx_point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring.extend([c1, c2])

        offspring = offspring[:self.population_size]

        """
        # Mutation
        """
        size = self.n_parameters
        offspring = [
            genome + self.rng.normal(0, .1, size=size) * (self.rng.random(size=size) < self.mutation_rate)
            for genome in offspring
        ]
        self.population = offspring
        self.population[:len(best)] = best

    def stop(self) -> bool:
        return self and False

