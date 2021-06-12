import numpy as np

from problems.problems import EllipsoidProblem, SphereProblem, SchwefelProblem, WeierstrassProblem
from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing
from optimizers.cmaes_crowding import CMA_crowding

SEED = 1202052400
REPETITIONS = 1
SAVED_ITERATION = 1000
POPULATION_SIZE = 100
DIMENSION = 100
SIGMA = 40.0
MEAN = 3
KAPPA_FACTOR = 1.5
SIGMA_FACTOR = 1/2


def main():
    optimizer = CMA(mean=MEAN * np.ones(DIMENSION),
                    sigma=SIGMA, population_size=POPULATION_SIZE)
    problem = SchwefelProblem(DIMENSION)
    print(f" evals    {type(problem).__name__} vlaue")

    evals = 0
    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = problem.evaluate(x)
            evals += 1
            solutions.append((x, value))
            if evals % 3000 == 0:
                print(f"{evals:5d}  {value:15.15f}")
        optimizer.tell(solutions)

        if optimizer.should_stop():
            print(f"{evals:5d}  {value:15.15f}")
            break


if __name__ == "__main__":
    main()
