import numpy as np

from a2_exp.lib import EAStrategy


class CMAES(EAStrategy):
    """
    CMA-ES using the CMA package (takes a while to import)
    """

    def __init__(self, n_parameters: int, population_size: int, seed: int):
        super().__init__(n_parameters=n_parameters, population_size=population_size)

        import cma

        options = {
            "seed": seed,
            "popsize": population_size,
            'verb_disp': 0,
        }
        self.cma = cma.CMAEvolutionStrategy([0.] * n_parameters, .5, options)

    def ask(self):
        return self.cma.ask()

    def tell(self, samples, scores):
        return self.cma.tell(samples, scores)

    def stop(self):
        # return self.cma.stop()
        return self and False


class GA(EAStrategy):
    """
    genetic algorithm
    """
    def __init__(self, n_parameters: int, population_size: int, seed: int, **kwargs):
        super().__init__(n_parameters=n_parameters, population_size=population_size)
        self.rng = np.random.default_rng(seed)
        self.population = [self.rng.normal(size=self.n_parameters) for _ in range(self.population_size)]
        self.crossover_rate = kwargs.get("crossover_rate", .2)
        self.mutation_rate = kwargs.get("mutation_rate", .1)
        self.mutation_magnitude = kwargs.get("mutation_magnitude", .1)
        self.sampling_size = kwargs.get("sampling_size", 3)

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
        ps = self.population_size
        assert len(samples) == len(scores) == ps
        indices = list(range(len(samples)))
        scoring = dict(zip(indices, scores))
        selected_i = [min(self.rng.choice(indices, self.sampling_size), key=scoring.get) for _ in range(ps)]
        selected = [samples[i] for i in set(selected_i)]
        pairs = [self._blend(*self.rng.choice(selected, 2)) for _ in range(ps)]
        new_generation = [self._mutate(x) for pair in pairs for x in pair]
        new_generation = self.rng.choice(new_generation, size=ps)
        assert len(new_generation) == ps, f'new gen size {len(new_generation)} should be {ps}'
        self.population = new_generation

    def stop(self):
        return self and False


class RandomStrat(EAStrategy):
    """
    produces seeds for random policies
    """
    def __init__(self, seed: int, n_parameters: int, population_size: int):
        assert n_parameters == 1
        super().__init__(n_parameters, population_size)
        self.rng = np.random.default_rng(seed)
        self.count = 0

    def ask(self):
        return self.rng.integers(0, 10_000_000, size=self.population_size)

    def tell(self, samples, scores):
        pass

    def stop(self):
        return self and False
