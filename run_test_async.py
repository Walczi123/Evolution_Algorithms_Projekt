import multiprocessing
import time

import numpy as np

from tests.test import Test

from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing
from optimizers.cmaes_crowding import CMA_crowding

from problems.problems import EllipsoidProblem, SphereProblem, \
    RosenbrockProblem, AckleyProblem, RastrginProblem, \
    GriewankProblem, WeierstrassProblem, SchwefelProblem

SEED = 12026021
MAX_ITERATION = 300000
REPETITIONS = 10
SAVED_ITERATION = 1000
POPULATION_SIZE = 100
DIMENSION = 40
SIGMA = 40.0
MEAN = 3
KAPPA_FACTOR = 1.5
SIGMA_FACTOR = 1/2


def generate_instances():
    result = []

    # EllipsoidProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=EllipsoidProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       problem=EllipsoidProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=EllipsoidProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # SphereProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=SphereProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       SphereProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=SphereProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # RosenbrockProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=RosenbrockProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       RosenbrockProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=RosenbrockProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # AckleyProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=AckleyProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       AckleyProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=AckleyProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # RastrginProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=RastrginProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       RastrginProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=RastrginProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # GriewankProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=GriewankProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       GriewankProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=GriewankProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # # WeierstrassProblem
    # result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
    #                    problem=WeierstrassProblem(DIMENSION), n_repetition=REPETITIONS,
    #                    name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    # result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
    #                    WeierstrassProblem(DIMENSION), n_repetition=REPETITIONS,
    #                    name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    # result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
    #                    problem=WeierstrassProblem(DIMENSION), n_repetition=REPETITIONS,
    #                    name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    # SchwefelProblem
    result.append(Test(optimizer=CMA(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=SchwefelProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_clearing(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE, kappa_factor=KAPPA_FACTOR, sigma_factor=SIGMA_FACTOR),
                       SchwefelProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Clearing", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))
    result.append(Test(CMA_crowding(mean=MEAN * np.ones(DIMENSION), sigma=SIGMA, population_size=POPULATION_SIZE),
                       problem=SchwefelProblem(DIMENSION), n_repetition=REPETITIONS,
                       name="CMAES_Crowding", seed=SEED, saved_iterations=SAVED_ITERATION, max_iteration=MAX_ITERATION))

    return result


def run_test(test):
    print("in: " + type(test.problem).__name__ + " " + test.name)
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
