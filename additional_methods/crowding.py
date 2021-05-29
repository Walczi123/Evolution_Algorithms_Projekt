import random
import itertools

def d(pair):
    return abs(pair[0] - pair[1])

def recomnbination(p1, p2):
    c1 = (p1 * 2 + p2) /3
    c2 = (p1 + p2 * 2) /3
    return (c1, c2)

def competition(pair):
    return random.choices(pair, k=1)[0]

def crowding(population):
    parents = population

    # pairing
    len_parents = len(parents)
    indexes = list(range(0, len_parents - 1))
    tmp = random.choice(list(itertools.permutations(indexes)))
    
    p_indexes = list()
    for i in range(int(len_parents/2)):
        p_indexes.append((tmp[i],tmp[i+1]))

    if len_parents%2:
        p_indexes.append((len_parents-1, None))

    # recomnbinations
    len_indexes = len(p_indexes)
    children = list()
    for i in range(len_indexes):
        c1, c2 = recomnbination(population[p_indexes[i][0]], population[p_indexes[i][1]])
        children.append(c1)
        children.append(c2)

    # competitions
    for i in range(len(p_indexes)):
        p1c1 = (population[p_indexes[i][0]], children[p_indexes[i][0]])
        p2c2 = (population[p_indexes[i][1]], children[p_indexes[i][1]])

        p1c2 = (population[p_indexes[i][0]], children[p_indexes[i][1]])
        p2c1 = (population[p_indexes[i][1]], children[p_indexes[i][0]])

        l = d(p1c1) + d(p2c2)
        r = d(p1c2) + d(p2c1)

        if l < r:
            population[p_indexes[i][0]] = competition(p1c1)
            population[p_indexes[i][1]] = competition(p2c2)
        else:
            population[p_indexes[i][0]] = competition(p1c2)
            population[p_indexes[i][1]] = competition(p2c1)

    print(population)

if __name__ == "__main__":
    crowding([1,2,3,4,5,6])