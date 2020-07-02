import time
import gurobipy
from gurobipy import GRB
import numpy as np


class GurobiSolver(object):
    def __init__(self):
        super(GurobiSolver, self).__init__()

    def solve(self, nodes_num, edges, verbose=True):
        if verbose:
            print(f"start solving by gurobi native ...")
        start_time = time.time()
        model = gurobipy.Model()

        # add constrainr to the solver
        nodes = model.addMVar(nodes_num, vtype=GRB.BINARY, name="nodes")
        for i in range(edges.shape[0]):
            model.addConstr(nodes[edges[i, 0]] + nodes[edges[i, 1]] <= 1, f"c{i}")

        # objective
        model.setObjective(sum(nodes), GRB.MAXIMIZE)
        model.optimize()

        if verbose:
            print(f"solve finished in {time.time() - start_time}")

        results = [var.X for var in model.getVars()]

        return results, time.time() - start_time


if __name__ == "__main__":
    nodes_num = 3
    edges = np.array([[0, 1], [0, 2], [1, 2],])

    solver = GurobiSolver()
    results, time_used = solver.solve(nodes_num=nodes_num, edges=edges)

    print(f"the result : {results} in {time_used} ")
