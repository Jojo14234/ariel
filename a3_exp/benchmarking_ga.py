import numpy as np

from a3_exp.lib import TARGET_POSITION
from a3_exp.strategies import GA, CMAES


def main():
    fitness = lambda xs: -np.sqrt(sum((b - a) ** 2 for a, b in zip(xs, TARGET_POSITION)))
    es = GA(3, 200, 42, .1, .2)
    es = CMAES(3, 14, 42)

    for gen in range(100):
        pop = es.ask()
        score = list(map(fitness, pop))
        es.tell(pop, score)
        print(f"{gen=} | mu: {sum(score)/len(score):.2f}, max: {max(score):.3f}, min{min(score):.3f}")



if __name__ == '__main__':
    main()
