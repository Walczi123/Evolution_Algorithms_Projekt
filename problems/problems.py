import numpy as np
from .problem import Problem


class EllipsoidProblem(Problem):

    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # print("ellipsoid problem")
        n = len(x)
        if len(x) < 2:
            raise ValueError("dimension must be greater one")
        return sum([(1000 ** (i / (n - 1)) * x[i]) ** 2 for i in range(n)])


class SphereProblem(Problem):
    # See: https://www.sfu.ca/~ssurjano/spheref.html
    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)
        re = sum(np.power(x[i], 2) for i in range(self.D))
        return re


class RosenbrockProblem(Problem):
    # See: https://www.sfu.ca/~ssurjano/rosen.html
    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)
        re = 0
        for i in range(self.D - 1):
            re += 100 * \
                np.power((np.power(x[i], 2) - x[i + 1]),
                         2) + np.power((x[i] - 1), 2)
        return re


class AckleyProblem(Problem):
    # See: https: // www.sfu.ca/~ssurjano/ackley.html
    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)

        # shift operation
        # x = x - 42.0969

        part1 = 0
        for i in range(self.D):
            part1 += np.power(x[i], 2)
        part2 = 0
        for i in range(self.D):
            part2 += np.cos(2 * np.pi * x[i])
        re = -20 * np.exp(-0.2 * np.sqrt(part1 / self.D)) \
            - np.exp(part2 / self.D) + 20 + np.e
        return re


class RastrginProblem(Problem):
    # See: https://www.sfu.ca/~ssurjano/rastr.html
    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)
        re = 0
        for i in range(self.D):
            re += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10
        return re


class GriewankProblem(Problem):
    # See: https://www.sfu.ca/~ssurjano/griewank.html
    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)
        part1, part2 = 0, 1
        for i in range(self.D):
            part1 += x[i] ** 2
            part2 *= np.cos(x[i] / np.sqrt(i + 1))
        re = 1 + part1 / 4000 - part2
        return re


class WeierstrassProblem(Problem):

    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)
        part1 = 0
        for i in range(self.D):
            for j in range(21):
                part1 += np.power(0.5, j) * np.cos(2 * np.pi *
                                                   np.power(3, j) * (x[i] + 0.5))
        part2 = 0
        for i in range(21):
            part2 += np.power(0.5, i) * np.cos(2 *
                                               np.pi * np.power(3, i) * 0.5)
        re = part1 - self.D * part2
        return re


class SchwefelProblem(Problem):
    #  See: https://www.sfu.ca/~ssurjano/schwef.html
    def __init__(self, dimension):
        super().__init__(dimension)

    def evaluate(self, x):
        # x = self.lb + x * (self.ub - self.lb)
        part1 = 0
        for i in range(self.D):
            part1 += x[i] * np.sin(np.sqrt(np.abs(x[i])))
        re = 418.9829 * self.D - part1
        return re
