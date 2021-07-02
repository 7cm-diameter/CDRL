if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray
    from pandas import DataFrame

    from cdrl.env import NArmBandit
    from cdrl.model import QLearner
    from cdrl.simulation import Result, run_block

    parser = ArgumentParser()
    parser.add_argument("--file", "-f")
    parser.add_argument("--alpha", "-a", type=float, default=0.1)
    parser.add_argument("--beta", "-b", type=float, default=1.)
    parser.add_argument("--weight", "-w", type=float, default=2.)
    args = parser.parse_args()

    NUMBER_OF_ARMS = 2
    BLOCKSIZE = 150

    probs = np.array([[1., 1.], [1., 0.], [0., 0.], [0.5, 0.5]])

    agent = QLearner(args.alpha, args.beta, NUMBER_OF_ARMS)
    env = NArmBandit(NUMBER_OF_ARMS)

    def run(prob: NDArray[np.float_]) -> Result:
        env.set_probs(prob)
        return run_block(agent, env, BLOCKSIZE)

    result = np.vstack(list(map(run, probs)))
    ret = DataFrame(result,
                    columns=[
                        "lprob", "rprob", "reward", "action", "lq", "rq", "lc",
                        "rc", "lp", "rp"
                    ])

    datadir = Path(__file__).absolute().parent.parent.joinpath("data")
    if not datadir.exists():
        datadir.mkdir()
    datapath = datadir.joinpath(args.file)

    ret.to_csv(datapath, index=False)
