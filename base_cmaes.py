import cma

if __name__ == "__main__":
    es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
    es.optimize(cma.ff.rosen)