import mujoco as mj
import numpy as np

from a2_exp.lib import EAPolicy


class SinePolicy(EAPolicy):
    """

    output[i] = a_i1 * sin(b_i1 * t) + a_i2 * sin(b_i2 * t) + ...
    a_ij and b_ij are learned through evolution, t is the wall time of the simulation

    intuition is that no sensors are required for an animal that is very stable at rest (gecko),
    all that is needed is to move muscles in a sinusoidal rhythm, the frequency of which is learnt.

    """
    def __init__(self, in_features: int = 5, out_features: int = 8):
        self._a = np.zeros((in_features, out_features))
        self._b = np.zeros((in_features, out_features))

    def n_parameters(self) -> int:
        return self._a.size + self._b.size

    def bind(self, genome: np.ndarray):
        self._a = genome[:self._a.size].reshape(self._a.shape)
        self._b = genome[-self._b.size:].reshape(self._b.shape)
        return self

    def __call__(self, mj_model: mj.MjModel, mj_data: mj.MjData):
        return (self._a * np.sin(self._b * mj_data.time)).sum(axis=0)
