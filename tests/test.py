from optimizers.cmaes import CMA
from problems.problem import Problem


class Test:
    def __init__(self,  optimizer: CMA, problem: Problem, n_repetition, seed, name, saved_iterations=3000, max_iteration=500000):
        self.optimizer = optimizer
        self.problem = problem
        self.n_repetition = n_repetition
        self.seed = seed - 1
        self.name = name
        self.saved_iterations = saved_iterations
        self.max_iteration = max_iteration

    def compute(self):
        result = []
        result.append(
            f"seed\t{self.seed}\tpoultation_size\t{self.optimizer.base_population_size}\tmean\t{self.optimizer.base_mean[0]}\tsigma\t{self.optimizer.base_sigma}\n")
        result.append(f"evals\t{type(self.problem).__name__} vlaue\n")

        evals = 0
        while True:
            solutions = []
            for _ in range(self.optimizer.population_size):
                x = self.optimizer.ask()
                value = self.problem.evaluate(x)
                evals += 1
                solutions.append((x, value))
                if evals % self.saved_iterations == 0 or evals == 1:
                    result.append(f"{evals:5d}\t{value:15.15f}\n")
            self.optimizer.tell(solutions)

            if self.optimizer.should_stop() or evals > self.max_iteration:
                result.append(f"{evals:5d}\t{value:15.15f}\n")
                s = "("
                for v in x:
                    s += str(v)+","
                s = s[:-1] + ")"
                result.append(f"last result:\t{s}\n")
                break
        return result

    def start(self):
        # results = []
        s = self.seed
        for _ in range(self.n_repetition):
            if self.seed is not None:
                self.optimizer.reset()
                self.seed = self.seed + 1
                self.optimizer.set_seed(self.seed)
            result = self.compute()
            self.save_to_file(result, "results/" +
                              type(self.problem).__name__ + "_" +
                              self.name + "_" +
                              str(self.seed) + ".txt")
        self.seed = s

    def save_to_file(self, results, file_path):
        f = open(file_path, "w")
        for r in results:
            f.write(r)
        f.close
