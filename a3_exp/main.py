import os
import time
from concurrent.futures import ProcessPoolExecutor as PPool
from typing import NamedTuple, Dict, Callable, Optional

import mujoco as mj
import numpy as np

from a3_exp.lib import Experiment, SPAWN_RUGGED
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
        # quit_map = {0: -7, 5: -5.4, 10: -5, 15: -4.7, 20: -4.2, 25: -3.9, 35: -3.5}
        quit_map = {0: -5, 5: -4.0, 10: -3.9, 15: -3.8, 20: -3.6, 25: -3.4, 35: -3.0}
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
            res = cls.evaluate(**sim_kw, policy=factory().bind(genomes[amax]))
            print(scores[amax], res)

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
            world = lambda spec: self.spec_to_olympic_world(spec, SPAWN_RUGGED)

        best_scores, best_graphs = [], []

        with DummyPool() as pool:
            opool, ipool = DummyPool(), pool
            for i_og in range(1, config.outer_generations + 1):
                t = time.perf_counter()
                inner_kw['n_generations'] = min(inner_kw['n_generations'] + 2 * (i_og % 2 == 0), 40)
                inner_kw['sim_duration'] = min(inner_kw['sim_duration'] + 1 * (i_og % 2 == 0), 50)
                print(f"outer gen {i_og:03} | fd={fd_count()} | starting {repr_now()}")
                print(repr_kw(inner_kw))
                genomes = es.ask() # BODY GENOMES
                graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
                b_specs = list(map(self._graph_to_mj_spec, graphs))
                b_spec_xml = list(map(mj.MjSpec.to_xml, b_specs))
                w_specs = list(map(world, b_specs))
                w_spec_xml = list(map(mj.MjSpec.to_xml, w_specs))
                w_specs = [mj.MjSpec.from_string(xml) for xml in w_spec_xml]

                models = [w.compile() for w in w_specs]
                g_str = [self._graph_to_string(g) for g in graphs]

                futures = [
                    opool.submit(self.run_inner, mj_model=model, **inner_kw, pool=ipool, og=i_og, op=i)
                    for i, model in enumerate(models)
                ]
                gen_scores, gen_genomes, times = zip(*[fut.result() for fut in futures])
                i = argmax(gen_scores)
                sc = [
                    type(self).evaluate(
                        models[i],
                        NNPolicy.from_model(models[i])().bind(gen_genomes[i]),
                        self.fitness,
                        sim_time=inner_kw['sim_duration'],
                    ) for _ in range(10)
                    ]
                print(f"given: {gen_scores[i]:.5f}")
                print(" ".join(f"{x:.5f}" for x in sc))
                exit()

                for i, score in enumerate(gen_scores):
                    print(f"outer gen {i_og:03} {i:02} | score={score:.2f} | {times[i]}")

                es.tell(genomes, gen_scores)
                obj = dict(
                    i_og=i_og,
                    body_genomes=genomes,
                    scores=gen_scores,
                    body_graphs=g_str,
                    body_specs=b_spec_xml,
                    world_specs=w_spec_xml,
                    brain_genomes=gen_genomes,
                    sim_duration=inner_kw['sim_duration'],
                    n_gen_inner=inner_kw['n_generations'],
                )
                self.save(f"{name}_{i_og:03}", obj)
                amax = argmax(gen_scores)
                best_scores.append(gen_scores[amax])
                best_graphs.append(g_str[amax])
                mu, max_ = sum(gen_scores) / len(gen_scores), gen_scores[amax]
                max_g = max(best_scores)
                print(
                    f"outer gen {i_og:03} | fin, mu: {mu:.2f}, genmax {max_:.2f} vs cmax {max_g:.2f} | {repr_now()}"
                )
                elapsed = time.perf_counter() - t
                print(f"outer gen {i_og:03} took {elapsed:.2f}s ({int(elapsed / 60)}m)")

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
        self.init_nde_hpd(16)
        objs = [self.load(f"{name}_{i:03}") for i in range(1, 400) if self.exists(f"{name}_{i:03}")]
        obj_t = {k: [obj[k] for obj in objs] for k in objs[0]}

        ig = argmax(list(map(max, obj_t['scores'])))
        ip = argmax(obj_t['scores'][ig])
        model = mj.MjSpec.from_string(obj_t['world_specs'][ig][ip]).compile()
        policy = NNPolicy.from_model(model)().bind(obj_t['brain_genomes'][ig][ip])

        sd = obj_t['sim_duration'][ig]
        curr_score = self.evaluate(model, policy, self.fitness, sim_time=sd)
        print(f"saved score: {obj_t['scores'][ig][ip]:.3f} vs shown: {curr_score:.3f}")
        # self.view(model, policy, self.fitness, sim_time=sd[ig])

    def confirm_results(self, name: str):
        # i_og
        # body_genomes
        # scores
        # body_graphs
        # body_specs
        # world_specs
        # brain_genomes
        # sim_duration
        # n_gen_inner
        self.init_nde_hpd(16)
        world = lambda spec: self.spec_to_olympic_world(spec, SPAWN_RUGGED)

        with PPool() as pool:
            for i in range(1, 400):
                if not self.exists(f"{name}_{i:03}"): break
                obj = self.load(f"{name}_{i:03}")
                graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in obj['body_genomes']]
                g_str = list(map(self._graph_to_string, graphs))
                print("graphs same?", g_str == obj['body_graphs'])
                b_specs = [self._graph_to_mj_spec(gr) for gr in graphs]
                b_specs_xml = list(map(mj.MjSpec.to_xml, b_specs))
                print("body specs same?", b_specs_xml == obj['body_specs'])
                w_specs = list(map(world, b_specs))
                w_spec_xml = list(map(mj.MjSpec.to_xml, w_specs))
                print("world specs same?", w_spec_xml == obj['world_specs'])

                # models = [w.compile() for w in w_specs]
                #
                # models =  [mj.MjSpec.from_string(w).compile() for w in obj['world_specs']]
                # policies = [NNPolicy.from_model(m)().bind(g) for m, g in zip(models, obj['brain_genomes'])]
                # sd = obj['sim_duration']
                # futs = [
                #     pool.submit(self.evaluate, m, p, self.fitness, sim_time=sd) for m, p in zip(models, policies)
                # ]
                # scores = [fut.result() for fut in futs]
                # print("=" * 20, f"{i:03}", "=" * 20)
                # print(f" ".join(f"{f:.3f}" for f in scores))
                # print(f" ".join(f"{f:.3f}" for f in obj['scores']))
                # print(f" ".join(f"{x - y:.3f}" for x, y in zip(scores, obj['scores'])))


if __name__ == '__main__':
    print("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED"))
    _main_config = ExpConfig(
        nde_seed=16,
        world=1,
        outer_strategy_cls="CMA",
        outer_strat_kw=dict(seed=16),
        # outer_strat_kw=dict(seed=16, mutation_rate=.1, crossover_rate=.7, tournament_size=3),
        outer_population=3,
        outer_generations=400,
        inner_strategy_cls="CMA",
        inner_strat_kw=dict(seed=16),
        # inner_strat_kw=dict(seed=16, mutation_rate=.1, crossover_rate=.7, tournament_size=10),
        inner_population=5,
        inner_generations=5,
        inner_policy_cls="NNPolicy",
        sim_duration=10,
        sim_steps_per_cycle=10,
    )
    MainExperiment().run(name='cma_cma_satJSX', config=_main_config)
    # MainExperiment().confirm_results("cma_cma_satJS")

