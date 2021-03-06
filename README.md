# README
## LET’S PLAY LEGO!
**Project Description**
Our project is to generate a LEGO .ldr file through an input image, to describe and show how to assemble the LEGO bricks to imitate the image.

**Code Structure**
Our project code is basically under the lego-solver project which contains a lot of modules performing distinct functions. For our solver, the code is under the directory of “/solvers”, but utilizing other modules’ functions especially being based on the “brick_modeling”, which provides fundamental operators and transformations as well as modeling methods.

**List of File Types**
File types listed below are needed in our project:
	* ._ldr_ - generated by lego studio, provides a placement of LEGO bricks.
	* _.pkl_ - Python dict dumped file, gives the information of a connected graph.
	* _.mzn_ - Minizinc solver instruction file, describes conditional constraints.
	* _.col_ - need to be downloaded from the lego studio website, describes the features of a specific LEGO brick.

**Main Code Structure**
_gen_sketch_placement.py_ constructs the potential placement graph for the solver to select from. The potential placement graph is generated from the brick we pre-selected by their ids.

_adjacency_graph.py_ converts potential placement graph into an undirected graph where there are two kinds of edges: collision edge and connection edge. The output graph will be dumped into a .pkl file.

_get_sketch.py_ benefits from the last step, parsing the .pkl file and inject parameters to the mini zinc solver, in order to find an optimal solution. The output of the procedure is a .ldr file which can be visualized by lego studio or our _main.py_ under the directory _lego-solver_.

**Todo List**
	- [ ] Currently we found out that the color layering is fully manual. This is because when we pass the parameters of the adjacency graph into the solver and get the solution, the node can’t be mapped back, which means we don’t the original color of a particular node. So we divide the image into regions based on different colors. ::This can be improved if we can map the nodes from the solution set to those of the potential node set.::

	- [ ] Besides the problem described above, we also figured out that the maximum object is tricky and trivial, somewhat greedy. ::Our next step is about to add some parameters to control this maximum formula. By comparing different performance among different parameters, the solution can be more optimal.::
