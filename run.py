from problems.problems import Sphere
from optimizers.cmaes import CMAES

if __name__ == "__main__":
    TaskProb = Sphere(50, -50, 50)
    Task = CMAES(TaskProb, 1000)
    Task.run()
