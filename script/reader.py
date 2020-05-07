import numpy as np

def read_edge_data_file(filename):
    with open(filename, "r") as fp:
        lines = [line.strip() for line in fp.readlines()]
    ind = lines.index("E")
    num_edges = int(lines[ind + 1])
    edges = list(map(
        lambda l: list(map(int, l.split(" "))),
        lines[ind + 2: ind + 2 + num_edges]
    ))
    return np.array(edges)


def read_out_data_file(filename):
    strings_to_ints = lambda it: list(map(float, it))
    matrices = []
    with open(filename) as fp:
        item_num = int(fp.readline().strip())
        for i in range(item_num):
            rows, cols = map(int, fp.readline().strip().split(" "))
            lines = [fp.readline() for _ in range(rows)]
            points = [strings_to_ints(line.strip().split(" ")) for line in lines]
            matrices.append(points)
    return np.array(matrices)