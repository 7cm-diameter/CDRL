import numpy as np
from numpy import typing as npt

from cdrl.env import NArmBandit
from cdrl.model import Agent

Result = npt.NDArray[np.float_]


def _step(agent: Agent, env: NArmBandit) -> Result:
    q_value = agent.q_value()
    curiosity = agent.curiosity()
    reward = env.reward()
    prob = agent.pi()
    action = agent.act(prob)
    agent.update(reward, action)
    ret = np.array([reward[action], action]).astype(np.float_)
    return np.hstack((env.probs, ret, q_value, curiosity, prob))


def run_block(agent: Agent, env: NArmBandit, n: int) -> Result:
    return np.vstack(list(map(lambda _: _step(agent, env), range(n))))
