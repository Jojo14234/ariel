import os
from concurrent.futures import ProcessPoolExecutor as PPool
from datetime import datetime
from pathlib import Path
from typing import Optional

import mujoco as mj
import numpy as np

from a3_exp.lib import Experiment
from a3_exp.policies.sine_policy import CPGPolicy
from a3_exp.strategies import CMAES
from a3_exp.utils import DummyPool, timeit

mj.set_mjcb_control(None)  # DO NOT REMOVE

now = lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
get_kw = (lambda **k: {"ip": 100, "ig": 100, "st": 10, "nc": 10, "seed": 42} | k)
repr_kw = (lambda d: " ".join(f"{k}_{v}" for k, v in d.items()))
get_fd = lambda: 'nan' if not (p := Path(f"/proc/{os.getpid()}/fd")).exists() else len(os.listdir(p))


class MainExperiment(Experiment):

    @classmethod
    def _inner_loop(
        cls,
        mj_model: mj.MjModel,
        ip: int,
        ig: int,
        st: int,
        nc: int,
        seed: int,
        pool: Optional[PPool] = None,
        quiet: bool = False
    ):
        pool = pool or DummyPool()
        cpg_factory = lambda: CPGPolicy(out_features=mj_model.nu)
        # cpg_factory = lambda: NNPolicy(len((_d:=mj.MjData(mj_model)).qpos) + len(_d.qvel), out_features=mj_model.nu)
        n_params = cpg_factory().n_parameters()
        es = CMAES(n_params, ip, seed)
        ekw = {"fitness": cls.fitness, "n_steps_per_cycle": nc, "sim_time": st}
        _q = quiet
        best_scores = []
        best_policies = []
        quit_map = {5: -5.4, 10: -5, 20: -4.5, 25:-4}

        for i_gen in range(ig):
            genomes = es.ask()
            policies = [cpg_factory().bind(g) for g in genomes]
            futures = [pool.submit(cls.evaluate, mj_model=mj_model, policy=policy, **ekw) for policy in policies]
            scores = [fut.result() for fut in futures]
            es.tell(genomes, [-f for f in scores])
            argmax = max(range(len(scores)), key=scores.__getitem__)
            best_scores.append(scores[argmax])
            best_policies.append(policies[argmax])
            fd_count = get_fd()
            # if not (i_gen % 5 or _q):
            #     print(f"inner {fd_count=} | {i_gen} | {best_scores[-1]:.2f} | {max(best_scores):.2f} | {now()}")

            if max(best_scores) < quit_map.get(i_gen, -7):
                print(f"early quitting {i_gen} ... max={max(best_scores):.2f}, min={min(best_scores):.2f}")
                break

        # return best_scores + [best_scores[-1]] * (ig - len(best_scores))
        argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        print(f"inner {fd_count=} | fin | {best_scores[argmax]:.2f} | {now()}")
        return best_scores[argmax], best_policies[argmax]

    def _outer_loop(self):
        print(f"outer | starting...")
        op, og = 14, 30
        ikw = get_kw(ig=20, ip=80, quiet=False)
        es = CMAES(64 * 3, op, 42)


        best = [] # score, genome, graph, model, policy

        with PPool() as pool:
            o_pool, i_pool = DummyPool(), pool
            # o_pool, i_pool = pool, None

            for i_gen in range(og):
                print(f"outer | gen {i_gen} | submitting... | {now()}")
                genomes = es.ask()
                graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
                models = [self.spec_to_olympic_world(self._graph_to_mj_spec(g)) for g in graphs]
                futures = [o_pool.submit(self._inner_loop, m, **ikw, pool=i_pool) for m in models]
                # scores_2d = [fut.result() for fut in futures]
                # self.save(f"scores_og_{i_gen}", scores_2d)
                scores, policies = zip(*[fut.result() for fut in futures])
                es.tell(genomes, [-f for f in scores])
                argmax = max(range(len(scores)), key=scores.__getitem__)
                best.append(tuple(l[argmax] for l in (scores, genomes, graphs, models, policies)))

                print(f"outer | gen {i_gen} | max={best[-1][0]:.2f} min={min(scores):.2f} | {now()}")

        # best_scores, *_ = zip(*best)
        # argmax = max(range(len(best_scores)), key=best_scores.__getitem__)
        # return best_models[argmax], best_policies[argmax]

    def main(self):
        with timeit("outer loop"):
            # model, policy = self._outer_loop()
            self._outer_loop()

        # input("ready?")
        # self.view(model, policy, self.fitness)

    def random_example(self):
        genotype = list(np.random.default_rng(42).random((3, 64)).astype(np.float32))
        graph = self._genotype_to_graph(genotype)
        mj_model = self.spec_to_olympic_world(self._graph_to_mj_spec(graph))
        self.view(mj_model, lambda _, d: d.ctrl, self.fitness)

    def compare_mp_speeds(self):
        flat = self
        n = os.cpu_count() - 4
        with PPool(max_workers=n) as pool:
            with timeit(f"warmup"):
                kw = get_kw(ip=n * 2, ig=10)
                self._inner_loop(flat, **kw, quiet=True, pool=pool)

            # with timeit(f"{n}x seq with pool"):
            #     for _ in range(n):
            #         self._inner_loop(flat, **kw, quiet=True, pool=pool)

            with timeit(f"{n}x pool with seq"):
                futures = [pool.submit(self._inner_loop, flat, **kw, quiet=True) for _ in range(n)]
                _ = [fut.result() for fut in futures]

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

        # 10/05

        ## mp speed
        warmup took 9.84s
        12x seq with pool took 60.24s
        12x pool with seq took 51.31s

        warmup took 10.91s
        60x seq with pool took 610.30s
        60x pool with seq took 485.89s


        > gecko learn NN on Olympic
            > -4.15 was best within 20s 100ig

        > gecko CPG on Olympic

        outer | gen 8 | max fitness: -2.7616, minf=-5.823433627237052 | 2025-10-05 21:35:01
        Updating f393757..240d679
        """

    def random_outer_loop(self):
        rng = np.random.default_rng(42)
        ikw = get_kw(ig=20, ip=80, st=20, quiet=False)
        high_score = -4
        scores = []

        with PPool() as pool:
            for i in range(1, 1001):
                genome = rng.uniform(-4, 4, size=3*64)
                graph = self._genotype_to_graph(list(genome.reshape(3, 64).astype(np.float32)))
                model = self.spec_to_olympic_world(self._graph_to_mj_spec(graph))
                score, _ = self._inner_loop(model, **ikw, pool=pool)
                scores.append(score)
                print(f"{i} | score: {score:.2f} | mean: {sum(scores) / i:.2f}")
                if score > high_score:
                    high_score = score
                    obj = dict(genome=genome, graph=self._graph_to_string(graph))
                    self.save(f"best_{i}_{abs(score):.2f}", obj)
        #     for i_gen in range(1, 31):
        #         print(f"outer {i_gen} | starting ...")
        #         genomes = [rng.uniform(-i_gen, i_gen, size=3*64) for _ in range(10)]
        #         graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
        #         models = [self.spec_to_olympic_world(self._graph_to_mj_spec(g)) for g in graphs]
        #         futures = [o_pool.submit(self._inner_loop, m, **ikw, pool=i_pool) for m in models]
        #         scores, policies = zip(*[fut.result() for fut in futures])
        #         argmax = max(range(len(scores)), key=scores.__getitem__)
        #         best.append(tuple(l[argmax] for l in (scores, genomes, graphs, models, policies)))
        #         all_scores.extend(scores)
        #         print(f"outer {i_gen} | max={max(scores):.2f} mean: {sum(scores) / len(scores):.2f}")
        #         print(f"outer {i_gen} | max={max(all_scores):.2f} mean: {sum(all_scores) / len(all_scores):.2f}")
        #
        # bs, bg, *_ = zip(*best)
        # argmax = max(range(len(bs)), key=bs.__getitem__)
        # print(f"best score OAT: {bs[argmax]:.2f}, out of {len(all_scores)} rng")
        # self.save('all_scores_rng', all_scores)
        # self.save('best_genome', bg[argmax])

    def _debug(self):
        """
        population = 100
        scores = [-5.8] * 100

        es.ask() -> population + mutation + crossover

        es.tell() ->
        new_pop = population + samples
        new_scores = scores + new_scores
        indices = sorted(range(len(new_pop)), key=new_scores.__getitem__)[:len(population)]
        population = [new_pop[i] for i in indices]
        scores = [new_scores[i] for i in indices]
        """


if __name__ == '__main__':
    MainExperiment().random_outer_loop()
