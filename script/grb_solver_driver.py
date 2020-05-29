import os
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

for filename in auto_files:
    args = ["./gurobi_solver", filename, target_dir]
    print(" ".join(args))
    sp.run(args, check=True, stdout=sp.DEVNULL)