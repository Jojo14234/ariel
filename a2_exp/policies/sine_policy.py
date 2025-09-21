from typing import Literal

import mujoco as mj
import numpy as np

from a2_exp.lib import EAPolicy


class SinePolicy(EAPolicy):
    """

    output[i] = b_i + m_i1 * sin(f_i1 * t) + m_i2 * sin(f_i2 * t) + ...
    b_i, m_ij and f_ij are learned through evolution, t is the wall time of the simulation

    intuition is that no sensors are required for an animal that is very stable at rest (gecko),
    all that is needed is to move muscles in a sinusoidal rhythm, the frequency of which is learnt.

    frequency_opt: fixed (0), linear (1), exp (1)

    """
    def __init__(self, in_features: int = 5, out_features: int = 8, frequency_opt: Literal[0, 1, 2] = 0):
        self.b = np.zeros((out_features,))
        self.m = np.zeros((in_features, out_features))
        self.f = np.zeros((in_features, out_features))
        self._f2 = 10. ** np.arange(2, 2 - in_features, -1)

        assert frequency_opt in (0, 1, 2)
        self.is_fixed = frequency_opt == 0
        self.is_exp = frequency_opt == 2

    def n_parameters(self) -> int:
        return self.b.size + self.m.size + self.f.size * (1 - self.is_fixed)

    def bind(self, genome: np.ndarray):
        i, j = self.b.size, self.b.size + self.m.size
        self.b = genome[:i].reshape(self.b.shape)
        self.m = genome[i:j].reshape(self.m.shape)

        if self.is_fixed:
            self.f = self.f + self._f2.reshape((-1, 1))
        else:
            self.f = genome[j:].reshape(self.f.shape)

        return self

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        f = 10 ** self.f if self.is_exp else self.f
        return self.b + (self.m * np.sin(f * mj_data.time)).sum(axis=0)
