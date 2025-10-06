import os
from concurrent.futures import ProcessPoolExecutor as PPool
from datetime import datetime
from pathlib import Path
from typing import Optional

import mujoco as mj
import numpy as np
import pandas as pd
import torch

from a3_exp.lib import Experiment
from a3_exp.policies.sine_policy import CPGPolicy
from a3_exp.strategies import CMAES
from a3_exp.utils import DummyPool, timeit, repr_now as now, repr_kw

mj.set_mjcb_control(None)  # DO NOT REMOVE

get_kw = (lambda **k: {"ip": 100, "ig": 100, "st": 10, "nc": 10, "seed": 42} | k)
get_fd = lambda: 'nan' if not (p := Path(f"/proc/{os.getpid()}/fd")).exists() else len(os.listdir(p))
argmax_ = lambda a: max(range(len(a)), key=a.__getitem__)

class ConsistencyExperiment(Experiment):

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
        factory = lambda: CPGPolicy(out_features=mj_model.nu)
        # factory = lambda: NNPolicy(len((_d:=mj.MjData(mj_model)).qpos) + len(_d.qvel), out_features=mj_model.nu)
        n_params = factory().n_parameters()
        es = CMAES(n_params, ip, seed)
        ekw = {"fitness": cls.basic_fitness, "n_steps_per_cycle": nc, "sim_time": st}
        _q = quiet
        best_scores = []
        best_policies = []
        # quit_map = {5: -5.4, 10: -5, 20: -4.5, 25:-4}
        quit_map = {5: .05, 10: .1, 20: .4}

        for i_gen in range(ig):
            genomes = es.ask()
            policies = [factory().bind(g) for g in genomes]
            futures = [pool.submit(cls.evaluate, mj_model=mj_model, policy=policy, **ekw) for policy in policies]
            scores = [fut.result() for fut in futures]
            es.tell(genomes, [-f for f in scores])
            argmax = argmax_(scores)
            best_scores.append(scores[argmax])
            best_policies.append(policies[argmax])
            fd_count = get_fd()
            if not _q:
                print(f"inner {fd_count=} | {i_gen} | {best_scores[-1]:.2f} | {max(best_scores):.2f} | {now()}")

            if max(best_scores) < quit_map.get(i_gen, -7):
                print(f"early quitting {i_gen} ... max={max(best_scores):.2f}, min={min(best_scores):.2f}")
                break

        # return best_scores + [best_scores[-1]] * (ig - len(best_scores))
        argmax = argmax_(best_scores)
        print(f"inner fd={get_fd()} | fin | {best_scores[argmax]:.2f} | {now()}")
        return best_scores[argmax], best_policies[argmax]

    def get_best_genome_from_random(self, pool: PPool, op: int):
        ikw = get_kw(ig=20, ip=80, quiet=True)
        rng = np.random.default_rng(42)
        genomes = [rng.normal(loc=0, scale=5, size=3 * 64) for _ in range(op)]
        graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
        models = [self.spec_to_simple_world(self._graph_to_mj_spec(g)) for g in graphs]
        scores, policies = zip(*(self._inner_loop(m, **ikw, pool=pool) for m in models))
        amax = argmax_(scores)
        return genomes[amax], graphs[amax], scores[amax], policies[amax]

    def main(self):
        with PPool() as pool:
            ge, gr, sc, po = self.get_best_genome_from_random(pool=pool, op=5)
            print(f"best scoring from random: {sc:.2f}")
            genomes = [ge] * 20
            graphs = [self._genotype_to_graph(list(g.reshape(3, 64).astype(np.float32))) for g in genomes]
            gstr = {self._graph_to_string(g) for g in graphs}
            if len(gstr) == 1:
                print("all graphs are the same\n" * 10)
            models = [self.spec_to_simple_world(self._graph_to_mj_spec(g)) for g in graphs]
            ikw = get_kw(ig=20, ip=80, quiet=True)
            scores, policies = zip(*(self._inner_loop(m, **ikw, pool=pool) for m in models))
            scores_df = pd.Series(scores).describe().to_frame("same graph").T
            print(scores_df.to_string())
            print(f"best scoring from random: {sc:.2f}")

    def compare_graphs(self):
        rng = np.random.default_rng(42)

        genome = rng.normal(size=(3, 64)).astype(np.float32)
        genome = list(genome.reshape(3, 64).astype(np.float32))
        graphs = [self._genotype_to_graph(genome) for _ in range(2)]
        graph_strs = [self._graph_to_string(g) for g in graphs]
        print(len(graph_strs[0]))


if __name__ == '__main__':
    for _ in range(10):
        torch.manual_seed(42)
        ConsistencyExperiment().compare_graphs()
