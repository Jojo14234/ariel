import time
from concurrent.futures import ProcessPoolExecutor
from typing import cast

from a2_exp.utils import timeit


import cma
import torch
import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np


from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


mj.set_mjcb_control(None)  # DO NOT REMOVE
with timeit("compiling spec & mj model"):
    MJ_SPEC = (w := SimpleFlatWorld()).spawn(gecko().spec, spawn_position=[0, 0, 2]) or w.spec
    MJ_MODEL = cast(mj.MjModel, MJ_SPEC.compile())
BRAIN_PARAMS = 48


def run_cma():
    es = cma.CMAEvolutionStrategy([0] * BRAIN_PARAMS, 0.5, {'popsize': 30})
    i = 0
    stop = False
    sf_pairs = []

    with ProcessPoolExecutor() as pool:
        while not stop:
            solutions = es.ask()
            fits = list(pool.map(evaluate, solutions))
            es.tell(solutions, [-f for f in fits])
            print(f"iter {i}: max fitness {max(fits)}")
            stop = es.stop() or i == 200
            i += 1
            sf_pairs.extend(zip(solutions, fits))

    best_sol = max(sf_pairs, key=lambda x: x[1])[0]
    evaluate(best_sol, view=True)

    # with open(f'best-sol.pkl', 'wb') as f:
    #     pickle.dump(best_sol, f)


def main():
    run_cma()


def evaluate(solution: np.ndarray, view: bool = False):
    _ng = torch.no_grad()
    _ng.__enter__()
    assert len(solution) == BRAIN_PARAMS
    mj_data = mj.MjData(MJ_MODEL)

    model_in = 5
    model_out = len(mj_data.ctrl)

    brain = torch.nn.Sequential(
        torch.nn.Linear(in_features=model_in, out_features=model_out, dtype=torch.float64),
    )
    n_parameters = sum(p.numel() for p in brain.parameters())
    assert n_parameters == BRAIN_PARAMS, f'{n_parameters=} != {BRAIN_PARAMS=}'

    i = 0
    for p in brain.parameters():
        j = i + p.numel()
        p.copy_(torch.tensor(solution[i: j], dtype=p.dtype).view_as(p))
        i = j

    viewer = view and mj_viewer.launch_passive(MJ_MODEL, mj_data).__enter__()

    sin_freq = torch.from_numpy(.1 ** np.arange(model_in)).type(torch.float64)
    for i in range(100):
        mj.mj_step(MJ_MODEL, mj_data, nstep=10)
        out = brain(sin_freq.mul(i).sin())
        mj_data.ctrl[:] = out.numpy()
        # mj_data.ctrl[:] = mj_data.ctrl + out.numpy()
        viewer and viewer.sync()
        viewer and time.sleep(1/60)

    viewer and viewer.__exit__(*(None,) * 3)
    _ng.__exit__(*(None,) * 3)

    xy = mj_data.geom('robot-core').xpos[:2]
    return xy[1]


if __name__ == "__main__":
    main()


