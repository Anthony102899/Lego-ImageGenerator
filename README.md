LEGO Solver
============

Usage
-------
To compile the program. First edit variable (`INC`) for the path to Eigen in `makefile` as appropriate (the root of Eigen installation).
Then,
```
make
```
To get the motion of a certain shape, find the data file in `data` directory, for example `disattached-square.txt`. Then,
```
make square_with_parallel_bar.png
```
Afterwards, find the corresponding image file in `img` folder.

Get 3d plot of the motion by running the script (Python 3, numpy and pyqtgraph required)
```
python3 script/grapher.py data/disattached-square.txt.out
```

Dependencies
------------
- `*.cpp`
  - Eigen
- `gurobi_solver.cpp`
  - [Gurobi (9.0)](https://www.gurobi.com/) (not open-source, you might need a proper license from them)
- `*.py`
  - Python 3
  - numpy, matplotlib, pyqtgraph