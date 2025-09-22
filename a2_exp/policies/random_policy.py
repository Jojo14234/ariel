from a2_exp.lib import EAPolicy

import mujoco as mj
import numpy as np

class RandomPolicy(EAPolicy):

    def __init__(self, delta: float = 0.05):
        self.rng = None
        self.delta = delta

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        assert self.rng
        num_joints = mj_model.nu
        hinge_range = np.pi / 2
        rand_moves = self.rng.uniform(low=-hinge_range,  # -pi/2
                                      high=hinge_range,  # pi/2
                                      size=num_joints)

        return mj_data.ctrl + rand_moves * self.delta

    def n_parameters(self) -> int:
        return 1

    def bind(self, genome: int):
        self.rng = np.random.default_rng(genome)
        return self
