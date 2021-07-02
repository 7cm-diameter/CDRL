import numpy as np

from cdrl.model import ArrayFloat, ArrayInt


class NArmBandit(object):
    def __init__(self, k: int):
        self._k = k
        self._probs: ArrayFloat = np.zeros(k)

    @property
    def probs(self) -> ArrayFloat:
        return self._probs

    def set_probs(self, probs: ArrayFloat):
        self._probs = probs

    def reward(self) -> ArrayInt:
        return np.random.binomial(n=1, p=self._probs)
