from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import typing as npt
from scipy.stats import beta

ArrayInt = npt.NDArray[np.int_]
ArrayFloat = npt.NDArray[np.float_]


def to_onehot(i: int, n: int) -> ArrayInt:
    return np.identity(n)[i]


def to_category(x: ArrayInt) -> np.int_:
    return np.argmax(x).astype(np.int_)


def _softmax(x: ArrayFloat, beta_: float) -> ArrayFloat:
    xmax = np.max(x)
    x_ = np.exp((x - xmax) * beta_)
    return x_ / np.sum(x_)


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def update(self, reward: ArrayInt, action: int):
        pass

    @abstractmethod
    def act(self, prob: ArrayFloat) -> int:
        pass

    @abstractmethod
    def pi(self) -> ArrayFloat:
        pass

    @abstractmethod
    def q_value(self) -> ArrayFloat:
        pass

    @abstractmethod
    def curiosity(self) -> ArrayFloat:
        pass


class QLearner(Agent):
    def __init__(self, alpha: float, beta_: float, k: int):
        self._k = k
        self._beta = beta_
        self._alpha = alpha
        self._q: ArrayFloat = np.zeros(k)
        self._curiosity: ArrayFloat = np.zeros(k)

    def q_value(self) -> ArrayFloat:
        return self._q

    def curiosity(self) -> ArrayFloat:
        return self._curiosity

    def update(self, reward: ArrayInt, action: int):
        action_ = to_onehot(action, self._k)
        td_err = np.float_(reward) - self._q
        self._q += self._alpha * td_err * np.float_(action_)

    def pi(self) -> ArrayFloat:
        return _softmax(self._q, self._beta)

    def act(self, prob: ArrayFloat) -> int:
        return np.random.choice(self._k, p=prob)


class HeirarchicalQLearner(QLearner):
    def __init__(self, alpha: float, beta_: float, w: float, k: int):
        super().__init__(alpha, beta_, k)
        self._w = w

    def update(self, reward: ArrayInt, action: int):
        action_ = to_onehot(action, self._k)
        td_err = np.float_(reward) - self._q
        self._q += self._alpha * td_err * np.float_(action_)
        self._curiosity += self._alpha * (np.abs(td_err) -
                                          self._curiosity) * np.float_(action_)

    def pi(self) -> ArrayFloat:
        x = self._q + self._w * self._curiosity
        return _softmax(x, self._beta)

    def act(self, prob: ArrayFloat) -> int:
        return np.random.choice(self._k, p=prob)
