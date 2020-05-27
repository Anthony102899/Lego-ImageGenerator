import os
import sys
import numpy as np
import subprocess as sp
import json

from matplotlib import pyplot as plt

if len(sys.argv) < 2:
    sys.argv += ["output/temp"]

target_dir = sys.argv[1]

try:
    jsons = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if file.endswith(".json")]
except FileNotFoundError:
    print("File not found, check the arguments: ", sys.argv)

def plot_json_curve(plt, json_file):
    with open(json_file) as fp:
        objSolPairs2d = json.load(fp)

    epsilons = []
    costs = []

    vertex_obj_vals = {
        "{}_{}".format(item["edge"], item["vertex"]): []
        for item in objSolPairs2d[0]["data"]
    }

    for i, pairs in enumerate(objSolPairs2d):
        epsilons.append(pairs["eps"])
        costs.append(pairs["cost"])
        for pair in pairs["data"]:
            obj_val = round(pair["obj"], 4)
            key = "{}_{}".format(pair["edge"], pair["vertex"])
            vertex_obj_vals[key].append(pair["obj"])
    
    def find_pivot(array):
        a1 = np.around(np.array(array[:-1]), 4)
        a2 = np.around(np.array(array[1:]), 4)
        index = np.array(np.abs(a1 - a2) < 1e-4).argmax()
        return index

    vertex_critical = {}
    keys = ["{}_{}".format(pair["edge"], pair["vertex"]) for pair in objSolPairs2d[0]["data"]]
    for key in keys:
        vertex_critical[key] = find_pivot(vertex_obj_vals[key])

    plot_name = os.path.split(json_file)[1].split(".")[0]
    # plt.plot(costs, vertex_obj_vals["1_1"], label=plot_name)


    return epsilons, costs, vertex_obj_vals, vertex_critical

parsed_tuples = []
for json_file in jsons:
    ret = plot_json_curve(plt, json_file)
    parsed_tuples.append(ret)

epsilons = parsed_tuples[0][0]
costs    = parsed_tuples[0][1]
_, _, vertex_obj_values_for_each_data, vertex_critical_for_each_data = zip(*parsed_tuples)
point_objectives_indices = [criticals["1_1"] for criticals in vertex_critical_for_each_data]
pts_costs = [costs[ind] for ind in point_objectives_indices]
pts_vals  = [objvals["1_1"][ind] for objvals, ind in zip(vertex_obj_values_for_each_data, point_objectives_indices)]
plt.scatter(pts_costs, pts_vals)
    

# plt.legend()
plt.xlabel("Max Cost")
plt.ylabel("Max L1 distance in velocity of vertex 1-1")
plt.savefig("out.png")