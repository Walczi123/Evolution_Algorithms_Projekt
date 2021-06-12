import numpy as np

from tests.test import Test

from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing

from problems.problems import EllipsoidProblem, SchwefelProblem


SEED = 1202052400
REPETITIONS = 1
SAVED_ITERATION = 100
POPULATION_SIZE = 1000
DIMENSION = 3
SIGMA = 40.0
MEAN = 3
KAPPA_FACTOR = 1.5
SIGMA_FACTOR = 1/2


if __name__ == "__main__":
    optimizer = CMA(mean=MEAN * np.ones(DIMENSION),
                    sigma=SIGMA, population_size=POPULATION_SIZE)
    problem = SchwefelProblem(DIMENSION)
    test = Test(optimizer, problem, n_repetition=10,
                name="Clearing_EllipsoidProblem", seed=123, max_iteration=100, saved_iterations=SAVED_ITERATION)
    test.start()
