import os
import sys
import subprocess as sp
import json

from matplotlib import pyplot as plt

if len(sys.argv) < 2:
    sys.argv += "output/temp"

target_dir = sys.argv[1]

jsons = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if file.endswith(".json")]

def plot_json_curve(plt, json_file):
    with open(json_file) as fp:
        objSolPairs2d = json.load(fp)

    epsilons = []
    costs = []

    vertexCosts = {
        "{}_{}".format(item["edge"], item["vertex"]): []
        for item in objSolPairs2d[0]["data"]
    }
    for pairs in objSolPairs2d:
        epsilons.append(pairs["eps"])
        costs.append(pairs["cost"])
        for pair in pairs["data"]:
            key = "{}_{}".format(pair["edge"], pair["vertex"])
            vertexCosts[key].append(pair["obj"])
    
    plot_name = os.path.split(json_file)[1].split(".")[0]
    plt.plot(costs, vertexCosts["2_1"], label=plot_name)

for json_file in jsons:
    plot_json_curve(plt, json_file)

plt.legend()
plt.xlabel("maxCost")
plt.ylabel("max L1 distance")
plt.savefig("out.png")