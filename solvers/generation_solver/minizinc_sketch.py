import os
import time
import numpy as np
import abc
import six
from minizinc import Solver, Model, Instance
from solvers.generation_solver.adjacency_graph import AdjacencyGraph

@six.add_metaclass(abc.ABCMeta)
class BaseSolver(object):
    def __init__(self):
        return
    @abc.abstractmethod
    def solve(self, brick_layout):
        return NotImplementedError

class MinizincSolver(BaseSolver):
    def __init__(self, model_file, solver_type):
        super(MinizincSolver, self).__init__()
        self.solver_type = solver_type
        self.solver = Solver.lookup(solver_type) # Find the MiniZinc solver configuration 
        self.model_file = model_file

    def solve(self, structure_graph, node_sd, node_area, node_weight, base_count, scale, verbose = True):
        if verbose:
            print(f"start solving by {self.solver_type} ...")
        start_time = time.time()
        model = Model()

        # Load model from file
        model.add_file(self.model_file)
        # Create an Instance
        instance = Instance(self.solver, model)
        # add data to the solver
        overlap_edges = np.array(structure_graph.overlap_edges)
        connect_edges = np.array(structure_graph.connect_edges)
        instance["base_count"]          = base_count
        instance["nodes_num"]           = len(structure_graph.bricks)
        instance["nums_edge_collision"] = len(structure_graph.overlap_edges)
        instance["nums_edge_connect"]   = len(structure_graph.connect_edges)
        instance["from_collision"]      = [int(edge + 1) for edge in overlap_edges[:, 0]]   if len(overlap_edges.shape) > 1 else []
        instance["to_collision"]        = [int(edge + 1) for edge in overlap_edges[:, 1]]   if len(overlap_edges.shape) > 1 else []
        instance["from_connect"]        = [int(edge + 1) for edge in connect_edges[:, 0]]   if len(connect_edges.shape) > 1 else []
        instance["to_connect"]          = [int(edge + 1) for edge in connect_edges[:, 1]]   if len(connect_edges.shape) > 1 else []
        instance["node_sd"]             = node_sd
        instance["node_area"]           = node_area
        instance["node_weight"]         = node_weight
        instance["sd_scale"]            = scale

        result = instance.solve()
        if result.status.has_solution():
        	selected_nodes = [1 if selected else 0 for selected in result['node']]
        else:
            selected_nodes = np.zeros(len(structure_graph.bricks))
            print("No solution found")

        time_used = round(time.time() - start_time, 2)
        if verbose:
            print(f"solve scale {scale} finished in {time_used}")

        return selected_nodes