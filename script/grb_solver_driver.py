import os
import multiprocessing
from itertools import product
import numpy as np
import subprocess as sp

print("Running script ", __file__, "current working dir: ", os.getcwd())

# read and populate the template
template_name = "multiple-beams"
template = open(f"data/template/{template_name}.txt").read()
out_dir  = "data/auto"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

frange = np.linspace(1, 10, 20)
irange = range(1, 20)
auto_files = []
# for i in frange:
    # mapping = {"x" : i}
for i in irange:
    a = i
    mapping = {
        "template_name": template_name,
        "n" : 3 + 2 * a,
        "m" : 2 + a,
        "num_anchors" : 1 + 2 * a,
        "points" : "\n".join(
            [f"{1 + 2 * x / a} 0 0" for x in range(a)] + [f"0 {1 + 9 * x / a} 0" for x in range(a)]
        ),
        "edges"   : "\n".join([f"{3 + x} {3 + x + a}" for x in range(a)]),
        "anchors" : "\n".join(
            [f"{3 + x} 1 {2 + x}" for x in range(a)] + [f"{3 + x + a} 0 {2 + x}" for x in range(a)]
        )
    }

    text = template.format_map(mapping)
    filename = "{template_name}-{num_anchors}.txt".format_map(mapping)
    path = os.path.join(out_dir, filename)
    auto_files.append(path) # gather the paths of the generated files for the next step
    with open(path, "w") as fp:
        fp.write(text)

# call gurobi_solver, write jsons to target dir
target_dir = f"output/{template_name}"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

sp.run(["make", "gurobi_solver"], shell=False)

eps_range = np.around(np.linspace(0.00, 0.5, 3), 6)
cost_range = np.around(np.linspace(1e-6, 1e-2, 2), 6)

from database import Adapter
import json

adapter = Adapter("./data/db/lp.json")
def run_solver(args_and_setting):
    args, setting = args_and_setting

    eid = adapter.generate_id(setting)
    res = adapter.get_experiment_by_id(eid)
    if res == []:
        sp.run(args, check=True, stdout=sp.DEVNULL)
        output_json = args[2]
        with open(output_json) as fp:
            result = json.load(fp)["result"]

        os.remove(output_json)

    else:
        result = res[0]["result"]

    return setting, res

for ind, filename in enumerate(auto_files):
    print(f"Running solver on {filename}, {ind}/{len(auto_files)}...")
    datafilename = os.path.split(filename)[1]
    args_and_settings = [
        (
            [
                "./gurobi_solver", 
                filename, 
                os.path.join(target_dir, f"{datafilename}-e-{eps}-c-{cost}.json"),
                str(eps), 
                str(cost)
            ],
            {
                "file": datafilename,
                "epsilon": eps,
                "cost": cost,
            }
        ) for eps, cost in product(eps_range, cost_range)
    ]

    with multiprocessing.Pool() as p:
        res = p.map(run_solver, args_and_settings)

    print(len(res))
    for setting, result in res:
        adapter.put_experiment(setting, result)
    print(len(adapter.conn.all()))

for root, _, files in os.walk(target_dir):
    for i, file in enumerate(files):
        if i % 300 == 0:
            print(i, len(files))