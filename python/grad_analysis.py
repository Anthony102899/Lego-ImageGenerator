from matplotlib import pyplot as plt
import torch
from torch import optim
import numpy as np
from reader import read_data_file

import sys

class GradientAnalyzer(object):
    def __init__(self, constraint: np.ndarray, vertices: np.ndarray, edges: np.ndarray):
        constraint_matrix = constraint[:, 6:]
        self.C : torch.Tensor = torch.tensor(constraint_matrix)
        self.vertices = vertices
        self.edges = edges


    def compute_length_for_edge(self, ind: int):
        p_ind = self.edges[ind, 0]
        q_ind = self.edges[ind, 1]

        p = self.vertices[p_ind, :]
        q = self.vertices[q_ind, :]

        return np.linalg.norm(p - q)

    def cost_func(self, x: torch.Tensor) -> float:
        states = x.reshape((-1, 6))
        energy = 0.0

        for i, vw in enumerate(states):
            i = i + 1

            v: torch.Tensor = vw[:3]
            w: torch.Tensor = vw[3:]

            length = self.compute_length_for_edge(i)
            mass = length 
            inertia = (1 / 12) * mass * length
            e_v = (1 / 2) * mass * v.norm()
            e_w = (1 / 2) * inertia * w.norm()
            energy += e_v + e_w
        
        return energy
    
    def objective_common(self, x: torch.Tensor) -> torch.Tensor:
        Cx = self.C.matmul(x)
        Cx_abs = Cx.abs()
        Cx_reshape = Cx_abs.reshape((-1, 6)).sum(dim=1)
        return Cx_reshape

    def objective_max(self, x: torch.Tensor) -> torch.Tensor:
        Cx_reshape = self.objective_common(x)
        f = torch.max(Cx_reshape)
        return torch.max(Cx_reshape)

    def objective_argmax(self, x: torch.Tensor) -> torch.Tensor:
        Cx_reshape = self.objective_common(x)
        f = torch.argmax(Cx_reshape)
        return f


    def find_vulnerable_constraint(self, max_cost: float, num_iter: int):
        hist = []
        for i in range(num_iter):
            x = torch.randn((self.C.size(1), 1), dtype=torch.double) / 1000
            x.requires_grad_(True)
            optimizer = optim.Adam([x], lr=0.0001)
            while self.cost_func(x) < max_cost:
                optimizer.zero_grad()
                obj = -self.objective_max(x)
                obj.backward()
                optimizer.step()
            
            max_ind = int(self.objective_argmax(x))
            obj = float(self.objective_max(x))

            hist.append((max_ind, obj, x.T))
            # print(max_ind, obj)

            if i % 50 == 0:
                print(i)

        return hist

if __name__ ==  "__main__":
    if len(sys.argv) < 2:
        object_name = "hard-ul-10"
    else:
        object_name = sys.argv[1]

    constraint_matrix = np.loadtxt(f"../data/matrix/{object_name}-constraint.csv", delimiter=",", dtype=np.double)
    vertices = read_data_file(f"../data/object/{object_name}.txt", "P")
    edges = read_data_file(f"../data/object/{object_name}.txt", "E")

    worker = GradientAnalyzer(constraint_matrix, vertices, edges)

    hist = worker.find_vulnerable_constraint(0.01, 1000)

    hist.sort(key=lambda t: t[1], reverse=True)
    # print(hist[:5])

    from collections import Counter
    print(Counter([h[0] for h in hist]).items())

    plt.boxplot([
        [x[1] for x in hist if x[0] == i]
        for i in range(4)
    ])
    plt.savefig("boxplot.png")
