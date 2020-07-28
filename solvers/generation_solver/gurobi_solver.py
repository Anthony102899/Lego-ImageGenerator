import time
import gurobipy
from gurobipy import GRB
import numpy as np


class GurobiSolver(object):
    def __init__(self):
        super(GurobiSolver, self).__init__()
        
    def solve(self, nodes_num, node_volume, overlap_edges, connect_edges, flag, verbose=True):
        print("start solving...")
        if verbose:
            print(f"start solving by gurobi native ...")
        start_time = time.time()
        model = gurobipy.Model()

        # add constraints to the solver
        nodes = model.addMVar(nodes_num, vtype=GRB.BINARY, name="nodes")
        overlap_num = len(overlap_edges)
        for i in range(overlap_num):
            model.addConstr(nodes[overlap_edges[i][0]] + nodes[overlap_edges[i][1]] <= 1, f"c{i}")
        for i in range(nodes_num):
            model.addConstr(nodes[i] <= sum([nodes[k] for k in connect_edges[i] if k]), f"c{i+overlap_num}")
        """
        TODO:
        change constraint: requires fully connected
            #filtered edge > #selected?
        """

        # objective  
        model.setObjective(sum([nodes[i]*node_volume[i]*flag[i] for i in range(nodes_num)]), GRB.MAXIMIZE)
        model.optimize()

        if verbose:
            print(f"solve finished in {time.time() - start_time}")

        results = [int(var.X) for var in model.getVars()]

        return results, time.time() - start_time


if __name__ == "__main__":
    nodes_num = 4
    overlapedges = np.array([[0, 3], [1, 3]])
    connectedges = np.array([[1, None], [0, 2], [1,3,None],[2]])
    nodes_areas = [1, 2, 2, 3]

    solver = GurobiSolver()
    results, time_used = solver.solve(nodes_num=nodes_num, node_volume=nodes_areas, overlap_edges=overlapedges, connect_edges=connectedges, flag=np.ones(nodes_num))

    print(f"the result : {results} in {time_used} ")
