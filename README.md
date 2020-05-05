LEGO Solver
============

Usage
-------
To compile the program. First edit variable (`LIBS`) for the path to Eigen in `makefile` as appropriate.
Then,
```
make
```
To get the motion of a certain shape, find the data file in `data` directory, for example `square_with_parallel.txt`. Then,
```
make square_with_parallel_bar.png
```
Afterwards, find the corresponding image file in `img` folder.