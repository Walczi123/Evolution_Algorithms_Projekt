import os

from optimizers.cmaes import CMA
from problems.problem import Problem


class Test:
    def __init__(self,  optimizer: CMA, problem: Problem, n_repetition, seed, name):
        self.optimizer = optimizer
        self.problem = problem
        self.n_repetition = n_repetition
        self.seed = seed - 1
        self.name = name

    def compute(self):
        result = []
        result.append(f" evals    {type(self.problem).__name__} vlaue\n")

        evals = 0
        while True:
            solutions = []
            for _ in range(self.optimizer.population_size):
                x = self.optimizer.ask()
                value = self.problem.evaluate(x)
                evals += 1
                solutions.append((x, value))
                if evals % 3000 == 0:
                    result.append(f"{evals:5d}  {value:15.15f}\n")
            self.optimizer.tell(solutions)

            if self.optimizer.should_stop():
                result.append(f"{evals:5d}  {value:15.15f}\n")
                break
        return result

    def start(self):
        # results = []
        s = self.seed
        for i in range(self.n_repetition):
            if self.seed is not None:
                self.optimizer.reset()
                self.seed = self.seed + 1
                self.optimizer.set_seed(self.seed)
            result = self.compute()
            self.save_to_file(result, "results/" +
                              self.name + "_" +
                              str(self.n_repetition) + "_" + str(self.seed) + ".txt")
        self.seed = s

    def save_to_file(self, results, file_path):
        f = open(file_path, "w")
        for r in results:
            f.write(r)
        f.close
