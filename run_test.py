import numpy as np

from tests.test import Test

from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing
from optimizers.cmaes_crowding import CMA_crowding

from problems.problems import EllipsoidProblem

if __name__ == "__main__":
    dim = 40
    optimizer = CMA_crowding(mean=3 * np.ones(dim), sigma=2.0)
    problem = EllipsoidProblem(dim)
    test = Test(optimizer, problem, n_repetition=1,
                name="Clearing_EllipsoidProblem", seed=123)
    test.start()
