LEGO Solver
============

## File organization
- `bricks_modeling` folder: to provide easy and common operations for LEGO bricks, e.g., transformation, abstraction
- `solvers` folder: solvers for LEGO rigidity analysis (to be implemented).
- `data` folder: to store LEGO models, and the ConnectivityGraph

## Dependency
- open3d
- trimesh
- scipy
- numpy
- sympy
- networkx
- pytorch

## Debugger
Everytime you run the program with `util.debugger.MyDebugger` instantiated, a folder will be created in the `debugger` folder under the root directory, with folder name `yyyy-mm-dd_hh-mm-ss_XXX`. Relevant run-time result and files will be stored in the folder.
