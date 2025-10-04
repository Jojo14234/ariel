import os

from concurrent.futures import ProcessPoolExecutor as PPool, as_completed
from datetime import datetime
from typing import Optional

import mujoco as mj
import numpy as np

from a3_exp.lib import Experiment
from a3_exp.strategies import CMAES
from a3_exp.utils import DummyPool, timeit

mj.set_mjcb_control(None)  # DO NOT REMOVE

now = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
get_kw = (lambda **k: {"ip": 100, "ig": 100, "st": 10, "nc": 10, "seed": 42} | k)
repr_kw = (lambda d: " ".join(f"{k}_{v}" for k, v in d.items()))

class MainExperiment(Experiment):

    @classmethod
    def _inner_loop(
        cls,
        mj_model: mj.MjModel,
        ip: int,
        ig : int,
        st: int,
        nc: int,
        seed: int,
        pool: Optional[PPool] = None,
        quiet: bool = False
    ):
        population_size = ip
        n_generations = ig
        sim_time = st
        n_steps_per_cycle = nc
        _q = quiet
        kw = dict(ip=ip, ig=ig, st=st, nc=nc, seed=seed)

        from a3_exp.policies.nn_policy import NNPolicy
        mj_data = mj.MjData(mj_model)
        in_features = len(mj_data.qpos) + len(mj_data.qvel)
        factory = lambda: NNPolicy(in_features=in_features, out_features=mj_model.nu)
        n_parameters = factory().n_parameters()
        _q or print(f"inner {os.getpid()} | starting... ({n_parameters=}) | {repr_kw(kw)} | {now()}")
        es = CMAES(n_parameters, population_size, seed)
        best_scores = []
        best_policies = []
        pool = pool or DummyPool()

        ekw = {"fitness": cls.basic_fitness, "n_steps_per_cycle": n_steps_per_cycle, "sim_time": sim_time}
        ekw = {"fitness": cls.fitness, "n_steps_per_cycle": n_steps_per_cycle, "sim_time": sim_time}

        for i_generation in range(n_generations):
            genomes = es.ask()
            policies = [factory().bind(g) for g in genomes]
            futures = [pool.submit(cls.evaluate, mj_model=mj_model, policy=policy, **ekw) for policy in policies]
            scores = [fut.result() for fut in futures]
            es.tell(genomes, [-f for f in scores])
            argmax = max(range(len(scores)), key=scores.__getitem__)
            best_scores.append(scores[argmax])
            best_policies.append(policies[argmax])
            i_generation % 5 or _q or print(f"inner | gen {i_generation}, max fitness: {best_scores[-1]:.4f} | {now()}")
            if es.stop():
                print(f'early stopping at generation {i_generation} ...')
                break

        argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        _q or print(f"inner {os.getpid()} | finished  | {now()}")
        return best_scores[argmax], best_policies[argmax]

    def _outer_loop(self, use_mp: bool = True):
        print(f"outer | starting...")
        _config = {
            "op": 14,
            "og": 30,
            "ip": 30,
            "ig": 30,
        } # 2 minutes for (*, 0, *, 5), so (*, *, *, *) is 2 * 30 * 30/5 = 6 hours
        """
        (*, 0, *, 5) took 2 minutes, implied 6 hours for all
        (*, 0, *, *) took 14 minutes, implied 7 hours for all
        """

        population_size = 14
        n_generations = 30
        n_parameters = 64 * 3
        es = CMAES(n_parameters, population_size, 42)
        inner_kwargs = {'population_size': 30, "n_generations": 30}
        best_models = []
        best_policies = []
        best_scores = []

        with (DummyPool, PPool)[use_mp]() as pool:
            for i_generation in range(n_generations):
                genomes = es.ask()
                graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
                models = [self._graph_to_mj_model(g) for g in graphs]
                print(f"outer | gen {i_generation} | submitting... | {now()}")
                futures = [pool.submit(self._inner_loop, model, **inner_kwargs) for model in models]
                scores, policies = zip(*[fut.result() for fut in futures])
                es.tell(genomes, [-f for f in scores])
                argmax = max(range(len(scores)), key=scores.__getitem__)
                print(f"outer | gen {i_generation} | max fitness: {max(scores):.4f} | {now()}")
                best_scores.append(scores[argmax])
                best_models.append(models[argmax])
                best_policies.append(policies[argmax])
                self.save(f"best_model_{i_generation}", {
                    "score": scores[argmax],
                    "model": models[argmax],
                    "policy": policies[argmax],
                })

        argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        return best_models[argmax], best_policies[argmax]

    def main(self, use_mp: bool = True):
        with timeit("outer loop"):
            model, policy = self._outer_loop(use_mp=use_mp)
        self.view(model, policy, self.fitness)

    def example(self):
        genotype = list(np.random.default_rng(42).random((3, 64)).astype(np.float32))
        graph = self._genotype_to_graph(genotype)
        mj_model = self._graph_to_mj_model(graph)
        self.view(mj_model, lambda _, d: d.ctrl, self.fitness)

    def gecko_hyper(self):
        gecko_model = self._gecko_world()
        with PPool() as pool:
            futures = {
                pool.submit(
                    self._inner_loop,
                    mj_model=gecko_model,
                    population_size=pop_size,
                    n_generations=80,
                    pool=None,
                    sim_time=sim_time,
                    n_steps_per_cycle=n_steps_per_cycle,
                ): (n_steps_per_cycle, pop_size)
                for n_steps_per_cycle in [5, 10, 20, 50]
                for pop_size in [20, 50, 100]
                for sim_time in [10, 20, 60]
            }
            print(len(futures))
            scores = []
            for fut in as_completed(futures):
                scores.append(fut.result()[0])
                params = futures[fut]
                print(f"{params} | {scores[-1]:.3f}")

    def gecko_time_pooling(self):
        model = self._gecko_world()

        with PPool() as pool:
            for kw in get_kw(ig=1), get_kw(ig=1), get_kw(ig=5):
                with timeit(f"POOLED CMA {repr_kw(kw)}"):
                    self._inner_loop(model, **kw, pool=pool, quiet=True)

            kw = get_kw(ig=5)
            with timeit(f"SEQP1_ CMA {repr_kw(kw)}"):
                pool.submit(self._inner_loop, model, **kw, quiet=True).result()

        with timeit(f"SEQUENT CMA {repr_kw(kw)}"):
            self._inner_loop(model, **kw, quiet=True)

    def gecko_timestep(self):
        model = self._gecko_world()
        with (PPool, DummyPool)[1]() as pool:
            for nc in 200, 200, 100, 50, 20, 10:
                kw = get_kw(ip=3, ig=3, nc=nc)
                with timeit(f"POOLED CMA {repr_kw(kw)}"):
                    score, _ = self._inner_loop(model, **kw, pool=None)
                    # print(f"{repr_kw(kw)} | {score=:.3f}")

    def gecko_base(self):
        model = self._gecko_simple_flat()

        with PPool() as pool:
            for kw in get_kw(ig=400),:
                with timeit(f"POOLED CMA | {repr_kw(kw)}"):
                    score, policy = self._inner_loop(model, **kw, pool=pool)

        self.save('best_gecko400', policy._genome)
        print(f"final score: {score}")
        # input("ready?")
        # self.view(model, policy, self.fitness)

    def gecko_example(self):
        gecko_model = self._gecko_world()
        policy = self.load("best_gecko")
        # self.view(gecko_model, lambda _, d: d.ctrl, self.fitness, n_steps_per_cycle=20)
        self.view(gecko_model, policy, self.fitness)

    def supernotes(self):
        """
        # 10/04

        ## First double loop implementation takes ages and produces shit results
        implemented CMA -> CMA and after 1 hour of 8, nothing found which scored higher than -5.6 (random)
        1 hour is way too long to have shit performance - is this the fault of brain loop or shit bodies?

        details: op_14 og_30 ip_30 ig_30 took 1 hour for og(3/30),

        decision: double loop is too difficult do debug, focus on getting a good gecko first, without a good
        gecko we have no hope.

        ## Gecko experiment

        ### Base
        POOLED CMA attempt=1 | ip_100 ig_100 st_10 nc_10 took 281.41s
        best score -4.6350
        took 75 gens to break -5.

        ### timing
        POOL-16    CMA ip_100 ig_1 st_10 nc_10 seed_42 took 11.97s
        POOL-16    CMA ip_100 ig_1 st_10 nc_10 seed_42 took 2.57s
        POOL-16    CMA ip_100 ig_5 st_10 nc_10 seed_42 took 12.33s
        SEQUENTIAL CMA ip_100 ig_5 st_10 nc_10 seed_42 took 89.45s
        POOL-1-out CMA ip_100 ig_5 st_10 nc_10 seed_42 took 78.21s
        89 / 12 = 7x slower, we have 8 cores but 16 logical processors, so this is either pretty bad or ok.

        # 10/04  3AM
        spent too long profiling shit, still only have some shit ass gecko, and still want to do more profiling

        # 10/04 22:24

        on ripper now

        """

        # self.save("best_gecko", policy)
        # input("ready?")


if __name__ == '__main__':
    MainExperiment().gecko_base()
