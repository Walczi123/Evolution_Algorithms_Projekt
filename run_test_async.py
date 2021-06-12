import multiprocessing
import time

import numpy as np

from tests.test import Test

from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing
from optimizers.cmaes_crowding import CMA_crowding

from problems.problems import EllipsoidProblem

SEED = 1202052400
REPETITIONS = 2
DIM = 40
SIGMA = 2.0
MEAN = 3
KAPPA_FACTOR = 1.5
SIGMA_FACTOR = 1/2


def generate_instances():
    result = []

    # EllipsoidProblem
    result.append(Test(CMA(mean=MEAN * np.ones(DIM), sigma=SIGMA),
                       EllipsoidProblem(), n_repetition=REPETITIONS,
                       name="CMAES_EllipsoidProblem", seed=SEED))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIM), sigma=SIGMA, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       EllipsoidProblem(), n_repetition=REPETITIONS,
                       name="CMAES_Clearing_EllipsoidProblem", seed=SEED))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIM), sigma=SIGMA),
                       EllipsoidProblem(), n_repetition=REPETITIONS,
                       name="CMAES_Crowding_EllipsoidProblem", seed=SEED))

    return result


def run_test(test):
    print("in: " + test.name)
    test.start()


def run_tests():
    iterable = generate_instances()
    start_time = time.time()

    p = multiprocessing.Pool()
    p.map_async(run_test, iterable)

    p.close()
    p.join()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    run_tests()
