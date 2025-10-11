import mujoco as mj
import numpy as np

from ariel import log
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import OlympicArena

TARGET_POSITION = [5, 0, 0.5]

def simulate(mj_model: mj.MjModel, policy, fitness, sim_time: int = 20, n_steps_per_cycle: int = 10):
    mj_data = mj.MjData(mj_model)
    while mj_data.time < sim_time:
        mj.mj_step(mj_model, mj_data, nstep=n_steps_per_cycle)
        mj_data.ctrl = np.clip(policy(mj_model, mj_data), -np.pi / 2, np.pi / 2)

    return fitness(mj_model, mj_data)


def default_fitness(_: mj.MjModel, mj_data: mj.MjData):
    distance = np.sqrt(sum((b - a) ** 2 for a, b in zip(mj_data.geom('robot1_core').xpos, TARGET_POSITION)))
    return -distance


def main():
    log.setLevel("DEBUG")
    seeded_rng = np.random.default_rng(42)
    robot_spec = gecko().spec
    world = OlympicArena()
    world.spawn(robot_spec)

    mj_model = world.spec.compile()
    policy = lambda _, d: d.ctrl + seeded_rng.normal(size=d.ctrl.shape)
    fitness = simulate(mj_model, policy, default_fitness)
    print(f"{fitness=:.4f}")


if __name__ == '__main__':
    main()
