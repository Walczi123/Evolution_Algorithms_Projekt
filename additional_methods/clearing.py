import math


def d(a, b):
    len_a = len(a)
    assert len_a == len(b), "Dimensions must be equal."
    sum = 0
    for i in range(len_a):
        sum += (a[i] - b[i]) ** 2
    return math.sqrt(sum)


def clearing(population, sigma=2, kappa=2):
    s = sorted(population, key=lambda s: s[1])
    s.reverse()
    n = len(s)
    for i in range(n):
        if s[i][1] > 0:
            nbWinner = 1
            for j in range(i+1, n):
                if s[i][1] > 0 and d(s[i][0], s[j][0]) < sigma:
                    if nbWinner < kappa:
                        nbWinner += 1
                    else:
                        s[j] = (s[j][0], 0)
    s.sort(key=lambda s: s[1])
    return s


if __name__ == "__main__":
    tmp = list()
    for i in range(9):
        # number, fitness
        tmp.append(([i], abs(4 - i)))
    res = clearing(tmp, 2.5, 4)
    print(res)
