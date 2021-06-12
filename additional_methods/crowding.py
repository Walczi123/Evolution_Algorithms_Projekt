import random
import itertools
import copy
import math
import numpy as np

def d(pair):
    a,b = pair
    a = a[0]
    b = b[0]
    len_a = len(a)
    assert len_a == len(b), "Dimensions must be equal."
    sum = 0
    for i in range(len_a):
        sum += (a[i] - b[i]) ** 2
    return math.sqrt(sum)


# def recomnbination(p1, p2):
#     c1 = (p1 * 2 + p2) / 3
#     c2 = (p1 + p2 * 2) / 3
#     return (c1, c2)


def competition(pair):
    a = pair[0][1]
    b = pair[1][1]
    if pair[0][1] > pair[1][1]:
        return pair[1]
    return pair[0]


def crowding(population, previous_population):
    if previous_population is None:
        return population
    children = copy.deepcopy(population)
    parents = previous_population
    # pairing
    len_parents = len(parents)
    tmp_indexes = list(range(0, len_parents))
    random.shuffle(tmp_indexes)
    p_indexes = list()
    for i in range(int(len_parents/2)):
        p_indexes.append((tmp_indexes[2*i], tmp_indexes[(2*i+1)]))

    # if len_parents % 2:
    #     p_indexes.append((tmp[len_parents-1], None))

    # recomnbinations
    # len_indexes = len(p_indexes)
    # children = list()
    # for i in range(len_indexes):
    #     c1, c2 = recomnbination(population[p_indexes[i][0]], population[p_indexes[i][1]])
    #     children.append(c1)
    #     children.append(c2)

    
    # competitions
    for i in range(len(p_indexes)):
        p1c1 = (parents[p_indexes[i][0]], children[p_indexes[i][0]])
        p2c2 = (parents[p_indexes[i][1]], children[p_indexes[i][1]])

        p1c2 = (parents[p_indexes[i][0]], children[p_indexes[i][1]])
        p2c1 = (parents[p_indexes[i][1]], children[p_indexes[i][0]])

        l = d(p1c1) + d(p2c2)
        r = d(p1c2) + d(p2c1)

        if l < r:
            children[p_indexes[i][0]] = competition(p1c1)
            children[p_indexes[i][1]] = competition(p2c2)
        else:
            children[p_indexes[i][0]] = competition(p1c2)
            children[p_indexes[i][1]] = competition(p2c1)

    if len_parents % 2:
        pc = (parents[tmp_indexes[len_parents-1]],
              children[tmp_indexes[len_parents-1]])
        children[tmp_indexes[len_parents-1]] = competition(pc)

    return children


if __name__ == "__main__":
    pop = [(np.array([1,  1,  1]), 1), (np.array([2,  2,  2]), 2), 
            (np.array([3,  3,  3]), 3), (np.array([4,  4,  4]), 4), (np.array([5,  5,  5]), 5)]
    r = [(np.array([6,  6,  6]), 6), (np.array([7,  7,  7]), 7), 
        (np.array([8,  8,  8]), 8), (np.array([9,  9,  9]), 9), (np.array([10,  10,  10]), 10)]
    print(crowding(pop, r))

