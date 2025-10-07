import time
from concurrent.futures import ProcessPoolExecutor as PPool
from typing import NamedTuple, Dict, Callable

import mujoco as mj
import numpy as np
from matplotlib import pyplot as plt

from a3_exp.lib import Experiment
from a3_exp.policies.nn_policy import NNPolicy, DoublePolicy
from a3_exp.policies.sine_policy import CPGPolicy, SinePolicy
from a3_exp.strategies import CMAES, GA, RandStrat
from a3_exp.utils import repr_now, argmax, fd_count, repr_kw

mj.set_mjcb_control(None)  # DO NOT REMOVE

P_MAP = {c.__name__: c for c in (NNPolicy, CPGPolicy, SinePolicy, DoublePolicy)}

class ExpConfig(NamedTuple):
    nde_seed: int
    world: int # 0 is simple, 1 is olympic

    outer_strategy_cls: str  # CMA, GA
    outer_strat_kw: Dict  # parameters, like seed
    outer_population: int
    outer_generations: int

    inner_strategy_cls: str  # CMA
    inner_strat_kw: Dict  # seed
    inner_population: int
    inner_generations: int

    inner_policy_cls: str  # CPGPolicy

    sim_duration: int  # 20s
    sim_steps_per_cycle: int  # 10


class SimResult(NamedTuple):
    graph_repr: str
    policy: str
    genome: np.ndarray


class MainExperiment(Experiment):

    @classmethod
    def run_inner(
        cls,
        mj_model: mj.MjModel,
        strategy_cls: str,
        strat_kw: Dict,
        policy_cls: str,
        population_size: int,
        n_generations: int,
        sim_duration: int,
        sim_steps_per_cycle: int,
        fitness: Callable,
        pool: PPool,
    ):
        factory = P_MAP[policy_cls].from_model(mj_model)
        n_parameters = factory().n_parameters()

        if n_parameters < 3:
            return [-6], [np.zeros(n_parameters)]

        es_cls = {"CMA": CMAES, "GA": GA}[strategy_cls]
        es = es_cls(n_parameters=n_parameters, population_size=population_size, **strat_kw)
        sim_kw = dict(mj_model=mj_model, fitness=fitness, sim_time=sim_duration, n_steps_per_cycle=sim_steps_per_cycle)
        # quit_map = {0: -7, 5: -5.4, 10: -5, 15: -4.7, 20: -4.2, 25: -3.9, 35: -3.5}

        bsc = []
        bg = []
        for gen in range(n_generations):
            genomes = es.ask()
            policies = [factory().bind(g) for g in genomes]
            t = time.perf_counter()
            futures = [pool.submit(cls.evaluate, policy=policy, **sim_kw) for policy in policies]
            scores = [max(-6, fut.result()) for fut in futures]
            print(f"i gen {gen} took {time.perf_counter() - t:.2f}s")
            es.tell(genomes, scores)
            amax = argmax(scores)
            bsc.append(scores[amax])
            bg.append(genomes[amax])
            print(f"i gen {gen} {policy_cls} | {bsc[-1]:.2f} | {max(bsc):.2f}, {max(bsc[-10:]):.2f}")
            # if gen % 5 == 0 and max(bsc[-10:]) - min(bsc) < gen / 10:
            #     break

        return bsc, bg

    def run(self, name: str, config: ExpConfig):
        self.init_nde_hpd(config.nde_seed)
        es_cls = {"CMA": CMAES, "GA": GA, "Rand": RandStrat}[config.outer_strategy_cls]
        es = es_cls(n_parameters=64 * 3, population_size=config.outer_population, **config.outer_strat_kw)
        inner_kw = dict(
            strategy_cls=config.inner_strategy_cls,
            strat_kw=config.inner_strat_kw,
            policy_cls=config.inner_policy_cls,
            population_size=config.inner_population,
            n_generations=config.inner_generations,
            sim_duration=config.sim_duration,
            sim_steps_per_cycle=config.sim_steps_per_cycle,
            fitness=(self.basic_fitness, self.fitness)[config.world],
        )
        world = (self.spec_to_simple_world, self.spec_to_olympic_world)[config.world]
        best_scores, best_graphs = [], []
        # n_generations 20 -> 40
        # sim_duration  10 -> 30

        with PPool() as pool:
            for i_og in range(1, config.outer_generations + 1):
                t = time.perf_counter()
                inner_kw['n_generations'] += 1 * (i_og % 5 == 0)
                inner_kw['sim_duration'] += 2 * (i_og % 5 == 0)
                print(f"outer gen {i_og:>2} | fd={fd_count()} | starting {repr_now()}")
                print(repr_kw(inner_kw))
                genomes = es.ask()
                graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
                models = [world(self._graph_to_mj_spec(g)) for g in graphs]
                g_str = [self._graph_to_string(g) for g in graphs]

                gen_scores = []
                gen_genomes = []
                for i_op, model in enumerate(models):
                    scores, inner_genomes = self.run_inner(mj_model=model, pool=pool, **inner_kw)
                    amax = argmax(scores)
                    gen_scores.append(scores[amax])
                    gen_genomes.append(inner_genomes[amax])
                    print(f"outer gen {i_og:>2} {i_op:>2} | score={scores[amax]:.2f}, ngen={len(scores)} | {repr_now()}")
                    # if scores[amax] > -4.2:
                    #     input("ready? ")
                    #     self.view(model, policy_cls.from_model(model)().bind(gen_genomes[-1]), self.fitness)


                es.tell(genomes, gen_scores)
                self.save(f"{name}_{i_og}_bodies", g_str)
                self.save(f"{name}_{i_og}_score_genome", (gen_scores, gen_genomes))
                amax = argmax(gen_scores)
                best_scores.append(gen_scores[amax])
                best_graphs.append(g_str[amax])
                mu, max_ = sum(gen_scores) / len(gen_scores), gen_scores[amax]
                max_global = max(best_scores)
                print(
                    f"outer gen {i_og:>2} | fin, mu: {mu:.2f}, genmax {max_:.2f} vs cmax {max_global:.2f} | {repr_now()}")
                elapsed = time.perf_counter() - t
                print(f"outer gen {i_og:>2} took {elapsed:.2f}s ({int(elapsed / 60)}m)")

        self.save(f"{name}_final", (best_scores, best_graphs))

    def run_gecko(self):
        inner_kw = dict(
            strategy_cls="GA",
            strat_kw=dict(seed=42, mutation_rate=.2, crossover_rate=.3),
            policy_cls="DoublePolicy",
            population_size=64 * 2,
            n_generations=100,
            sim_duration=20,
            sim_steps_per_cycle=10,
            fitness=self.basic_fitness,
        )
        with PPool() as pool:
            model = self.spec_to_simple_world(self.gecko_spec())
            res = {
                n: self.run_inner(mj_model=model, pool=pool, **inner_kw | dict(policy_cls=n))[0]
                for n in P_MAP
            }

        for k, v in res.items():
            print(k, max(v))
            print(" ".join(f"{v_:.2f}" for v_ in v))
            # plt.plot(v, label=k)
        # plt.legend()
        # plt.show()



if __name__ == '__main__':

    _test_config = ExpConfig(
        sim_duration=20,
        sim_steps_per_cycle=10,
        outer_population=30,
        outer_generations=100,
        inner_population=64 * 2,
        inner_generations=30,
        nde_seed=42,
        world=0,
        outer_strategy_cls="Rand",
        outer_strat_kw=dict(seed=42),
        inner_strategy_cls="CMA",
        inner_strat_kw=dict(seed=42),
        inner_policy_cls="NNPolicy",
    )

    _main_config = ExpConfig(
        nde_seed=42,
        world=1,
        outer_strategy_cls="GA",
        outer_strat_kw=dict(seed=42, mutation_rate=.2, crossover_rate=.7),
        outer_population=60,
        outer_generations=100,
        inner_strategy_cls="CMA",
        inner_strat_kw=dict(seed=42),
        inner_population=64,
        inner_generations=20,
        inner_policy_cls="CPGPolicy",
        sim_duration=10,
        sim_steps_per_cycle=10,
    )
    MainExperiment().run(name='_test', config=_test_config)
    # MainExperiment().run_gecko()
