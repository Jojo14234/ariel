import numpy as np

from a3_exp.main import MainExperiment
from a3_exp.strategies import GA


def main():
    f = lambda x: abs(x/10) * np.sin(x)
    fitness = lambda xs: sum(map(f, xs))

    es = GA(2, 200, 42, .7, .2)

    for gen in range(100):
        pop = es.ask()
        score = list(map(fitness, pop))
        es.tell(pop, score)
        print(f"{gen=} | mu: {sum(score)/len(score):.2f}, max: {max(score):.3f}, min{min(score):.3f}")
    print(fitness([12,12]))



if __name__ == '__main__':
    main()
