import time
from concurrent.futures import ProcessPoolExecutor as PPool
from typing import NamedTuple, Dict, Callable, Optional

import mujoco as mj
import numpy as np

from a3_exp.lib import Experiment, SPAWN_RUGGED, SPAWN_POS
from a3_exp.policies.nn_policy import NNPolicy, DoublePolicy
from a3_exp.policies.sine_policy import CPGPolicy, SinePolicy
from a3_exp.strategies import CMAES, GA, RandStrat
from a3_exp.utils import repr_now, argmax, fd_count, repr_kw, DummyPool

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
        pool: Optional[PPool] = None,
        og: int = 0,
        op: int = 0,
    ):
        pool = pool or DummyPool()
        factory = P_MAP[policy_cls].from_model(mj_model)
        n_parameters = factory().n_parameters()

        if n_parameters < 3:
            return [-6], [np.zeros(n_parameters)]

        es_cls = {"CMA": CMAES, "GA": GA}[strategy_cls]
        es = es_cls(n_parameters=n_parameters, population_size=population_size, **strat_kw)
        sim_kw = dict(mj_model=mj_model, fitness=fitness, sim_time=sim_duration, n_steps_per_cycle=sim_steps_per_cycle)
        # quit_map = {0: -5, 5: -4.8, 10: -4.5, 15: -4.2, 20: -4.0, 25: -3.5, 35: -3.0}
        quit_map = {0: -7, 5: -5.4, 10: -5, 15: -4.7, 20: -4.2, 25: -3.9, 35: -3.5}
        # quit_map = {}

        bsc = []
        bg = []
        for i_gen in range(n_generations):
            genomes = es.ask()
            policies = [factory().bind(g) for g in genomes]
            futures = [pool.submit(cls.evaluate, policy=policy, **sim_kw) for policy in policies]
            scores = [fut.result() for fut in futures]
            es.tell(genomes, scores)
            amax = argmax(scores)
            bsc.append(scores[amax])
            bg.append(genomes[amax])

            print(f"og:{og:>2} op:{op:>2} ig:{i_gen:>2} | {bsc[-1]:.2f} | {max(bsc):.2f} | {repr_now()}")
            if i_gen in quit_map and max(bsc) < quit_map[i_gen]:
                break
            # if gen % 5 == 0 and max(bsc[-10:]) - min(bsc) < gen / 10:
            #     break

        amax = argmax(bsc)
        return bsc[amax], bg[amax], repr_now()
        # return bsc[-1], bg[-1], repr_now()

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

        if config.world == 0:
            world = self.spec_to_simple_world
        else:
            world = lambda spec: self.spec_to_olympic_world(spec, SPAWN_POS)
        best_scores, best_graphs = [], []
        # n_generations 20 -> 40
        # sim_duration  10 -> 30

        with PPool() as pool:
            opool, ipool = DummyPool(), pool
            for i_og in range(1, config.outer_generations + 1):
                t = time.perf_counter()
                inner_kw['n_generations'] = min(inner_kw['n_generations'] + 2 * (i_og % 2 == 0), 40)
                inner_kw['sim_duration'] = min(inner_kw['sim_duration'] + 1 * (i_og % 2 == 0), 50)
                print(f"outer gen {i_og:>2} | fd={fd_count()} | starting {repr_now()}")
                print(repr_kw(inner_kw))
                genomes = es.ask() # BODY GENOMES
                graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
                specs = [self._graph_to_mj_spec(graph) for graph in graphs]
                models = [world(spec) for spec in specs]
                # models = [world(self._graph_to_mj_spec(g)) for g in graphs]
                g_str = [self._graph_to_string(g) for g in graphs]
                futures = [
                    opool.submit(self.run_inner, mj_model=model, **inner_kw, pool=ipool, og=i_og, op=i)
                    for i, model in enumerate(models)
                ]
                gen_scores, gen_genomes, times = zip(*[fut.result() for fut in futures])

                for i, score in enumerate(gen_scores):
                    print(f"outer gen {i_og:>2} {i:>2} | score={score:.2f} | {times[i]}")

                es.tell(genomes, gen_scores)
                self.save(f"{name}_{i_og}_bodies", g_str)
                self.save(f"{name}_{i_og}_score_genome", (gen_scores, gen_genomes))
                amax = argmax(gen_scores)
                best_scores.append(gen_scores[amax])
                best_graphs.append(g_str[amax])
                mu, max_ = sum(gen_scores) / len(gen_scores), gen_scores[amax]
                max_g = max(best_scores)
                print(
                    f"outer gen {i_og:>2} | fin, mu: {mu:.2f}, genmax {max_:.2f} vs cmax {max_g:.2f} | {repr_now()}"
                )
                elapsed = time.perf_counter() - t
                print(f"outer gen {i_og:>2} took {elapsed:.2f}s ({int(elapsed / 60)}m)")

        self.save(f"{name}_final", (best_scores, best_graphs))

    def run_gecko(self):
        inner_kw = dict(
            strategy_cls="CMA",
            strat_kw=dict(seed=42),
            policy_cls="NNPolicy",
            population_size=64,
            n_generations=50,
            sim_duration=20,
            sim_steps_per_cycle=10,
            fitness=self.basic_fitness,
        )
        model = self.spec_to_simple_world(self.gecko_spec())

        with PPool() as pool:
            score, genome, _ = self.run_inner(mj_model=model, pool=pool, **inner_kw)

        policy = NNPolicy.from_model(model)().bind(genome)
        self.view(model, policy, self.fitness)

    def view_results(self, name: str):
        graph_strs = []
        scores = []
        genomes = []
        for i in range(1, 40):
            if not self.exists(f"{name}_{i}_bodies"):
                print(f"{name}_{i}_bodies")
                break
            graph_strs.append(self.load(f"{name}_{i}_bodies"))
            s, g = self.load(f"{name}_{i}_score_genome")
            scores.append(s)
            genomes.append(g)

        graph = self._string_to_graph(graph_strs[1][1])
        # model = self.spec_to_olympic_world(self._graph_to_mj_spec(graph))
        model = self.spec_to_simple_world(self._graph_to_mj_spec(graph))
        genome = genomes[1][1]
        score = scores[1][1]
        policy = NNPolicy.from_model(model)().bind(genome)
        print(score)
        self.view(model, policy, self.fitness)

        inner_kw = dict(
            strategy_cls="CMA",
            strat_kw=dict(seed=42),
            policy_cls="NNPolicy",
            population_size=64,
            n_generations=20,
            sim_duration=10,
            sim_steps_per_cycle=10,
            fitness=self.fitness,
        )
        with PPool() as pool:
            self.run_inner(model, **inner_kw, pool=pool)
            self.run_inner(model, **inner_kw, pool=pool)
            self.run_inner(model, **inner_kw, pool=pool)


if __name__ == '__main__':

    _test_config = ExpConfig(
        sim_duration=20,
        sim_steps_per_cycle=10,
        outer_population=5,
        outer_generations=5,
        inner_population=60,
        inner_generations=10,
        nde_seed=42,
        world=1,
        outer_strategy_cls="GA",
        # outer_strat_kw=dict(seed=16),
        outer_strat_kw=dict(seed=16, mutation_rate=.1, crossover_rate=.7, tournament_size=3),
        inner_strategy_cls="CMA",
        inner_strat_kw=dict(seed=16),
        inner_policy_cls="NNPolicy",
    )

    _main_config = ExpConfig(
        nde_seed=16,
        world=1,
        outer_strategy_cls="CMA",
        outer_strat_kw=dict(seed=16),
        # outer_strat_kw=dict(seed=16, mutation_rate=.1, crossover_rate=.7, tournament_size=3),
        outer_population=8,
        outer_generations=400,
        inner_strategy_cls="CMA",
        inner_strat_kw=dict(seed=16),
        # inner_strat_kw=dict(seed=16, mutation_rate=.1, crossover_rate=.7, tournament_size=10),
        inner_population=60,
        inner_generations=20,
        inner_policy_cls="NNPolicy",
        sim_duration=10,
        sim_steps_per_cycle=10,
    )
    MainExperiment().run(name='ss_fri_aft_test', config=_test_config)
    # MainExperiment().view_results("SS_CONFIG_CMA2")
"""
og: 2 op:11 ig: 9 | -4.05 | -3.99 | 2025-10-07 19:13:43
og: 2 op:11 ig:10 | -3.33 | -3.33 | 2025-10-07 19:13:44
og: 2 op:11 ig:11 | -2.72 | -2.72 | 2025-10-07 19:13:46
"""
'''
outer gen 1: mu = 0.44
outer gen 2: mu = 0.75
outer gen 3: mu = 0.63
outer gen 4: mu = 0.47
outer gen 5: mu = 0.48
outer gen 6: mu = 0.66
outer gen 7: mu = 1.14
outer gen 8: mu = 0.72
outer gen 9: mu =
outer gen 32 score=9.052 1 |
'''
