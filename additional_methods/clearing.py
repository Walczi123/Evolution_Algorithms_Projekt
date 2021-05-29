import random
import itertools

def d(a, b):
    return random.choices(range(abs(a[0] - b[0])))[0]

def fitness(a):
    return a[1]

def clearing(population, sigma, kappa):
    s = sorted(population, key=fitness)
    s.reverse()
    n = len(s)
    for i in range(n):
        if fitness(s[i])>0 :
            nbWinner = 1
            for j in range(i+1, n):
                if fitness(s[i])>0 and d(s[i], s[j]) < sigma:
                    if nbWinner < kappa:
                        nbWinner += 1
                    else : 
                        #fitness(s[j]) = 0
                        s[j] = (s[j][0], 0)      
    print(s)

if __name__ == "__main__":
    tmp = list()
    for i in range(9):
        # number, fitness
        tmp.append((i, abs(4 - i)))
    clearing(tmp, 2.5, 4)
