import numpy as np

from tests.test import Test

from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing

from problems.problems import EllipsoidProblem

if __name__ == "__main__":
    dim = 40
    optimizer = CMA_clearing(mean=3 * np.ones(dim), sigma=2.0)
    problem = EllipsoidProblem()
    test = Test(optimizer, problem, n_repetition=10,
                name="Clearing_EllipsoidProblem", seed=123)
    test.start()
