import numpy as np

from problems.problems import EllipsoidProblem
from problems.problems import SphereProblem
from optimizers.cmaes import CMA
from optimizers.cmaes_clearing import CMA_clearing
from optimizers.cmaes_crowding import CMA_crowding


def main():
    dim = 40
    optimizer = CMA_crowding(mean=3 * np.ones(dim), sigma=2.0)
    problem = SphereProblem(dim)
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
