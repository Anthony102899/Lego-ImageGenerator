import os.path
import pickle
import solvers.generation_solver.sketch_util as util
from bricks_modeling.file_IO.model_writer import write_bricks_to_file

from solvers.generation_solver.minizinc_sketch import MinizincSolver
from solvers.generation_solver.precompute import PrecomputedModel
from util.debugger import MyDebugger


class SketchLayer:
    def __init__(self, filepath: str, solver: MinizincSolver):
        self._precompute_model: PrecomputedModel = pickle.load(open(filepath, "rb"))
        self._solver = solver
        self._selected_bricks_layer = []

    def generate_solutions(self):
        results = self._solver.solve(structure_graph=self._precompute_model.get_structure_graph(),
                                     node_sd=self._precompute_model.get_node_sd(),
                                     node_area=self._precompute_model.get_node_area(),
                                     node_weight=self._precompute_model.get_node_weight(),
                                     base_count=self._precompute_model.get_base_count())
        filtered_bricks = self._precompute_model.get_filtered_bricks()
        filtered_ldr_code = self._precompute_model.get_filtered_ldr_code()
        for i in range(self._precompute_model.get_base_count(), len(filtered_bricks)):
            if results[i] == 1:
                colored_brick = util.color_brick(filtered_bricks[i],
                                                 filtered_ldr_code[i],
                                                 rgb=False)
                self._selected_bricks_layer.append(colored_brick)

    def move_layer(self, layer: int):
        if self._precompute_model.get_background_bool():
            self._selected_bricks_layer = util.move_layer(self._selected_bricks_layer, layer + 1)
        else:
            self._selected_bricks_layer = util.move_layer(self._selected_bricks_layer, layer)

    def get_selected_bricks_layer(self):
        return self._selected_bricks_layer

    def get_base_bricks(self):
        return self._precompute_model.get_filtered_bricks()[:self._precompute_model.get_base_count()]

    def display_debug_info(self):
        print("-" * 9 + " Model Info " + "-" * 9)
        self._precompute_model.display_debug_info()
        print("-" * 30)


def get_sketch(file_names, layers):
    # num_of_layer = int(input("Please enter the number of layers: "))
    solver = MinizincSolver(os.path.dirname(__file__) + "/solve_sketch.mzn", "gurobi")
    selected_bricks = []
    # Todo: Handle background
    background_bool = 0
    num_of_layer = len(file_names)

    for i in range(num_of_layer):
        file_path = file_names[i]
        sketch_layer = SketchLayer(file_path, solver)
        sketch_layer.generate_solutions()
        sketch_layer.move_layer(layers[i])
        if i == 0:
            selected_bricks += sketch_layer.get_base_bricks()
        selected_bricks += sketch_layer.get_selected_bricks_layer()
        sketch_layer.display_debug_info()

    # Todo: Handle background color
    """if background_bool:
        background = os.path.dirname(__file__) + "/inputs/" + "back " + graph_name.split(".pkl")[0] + ".ldr"
        background = read_bricks_from_file(background)
        selected_bricks += util.move_brickset(background, background_rgb, 0, 0, ldr_color)"""
    result_name =  "test"  # input("Please enter the image name: ")

    debugger = MyDebugger(f"{result_name}")
    write_bricks_to_file(
        selected_bricks,
        file_path=debugger.file_path(f"{result_name}.ldr"))

    print("done!")



if __name__ == "__main__":
    num_of_layer = int(input("Please enter the number of layers: "))
    solver = MinizincSolver(os.path.dirname(__file__) + "/solve_sketch.mzn", "gurobi")
    selected_bricks = []
    # Todo: Handle background
    background_bool = 0

    file_names = []
    layers = []
    for i in range(num_of_layer):
        file_names.append(input(f"Please enter the {i + 1} file name: "))
        layers.append(int(input("Please enter the layer number of the above file: ")))

    for i in range(num_of_layer):
        file_path = os.path.dirname(__file__) + f"/precompute_models/{'_'.join(file_names[i].split('_')[:-1])}/" + \
                    file_names[i]
        sketch_layer = SketchLayer(file_path, solver)
        sketch_layer.generate_solutions()
        sketch_layer.move_layer(layers[i])
        if i == 0:
            selected_bricks += sketch_layer.get_base_bricks()
        selected_bricks += sketch_layer.get_selected_bricks_layer()
        sketch_layer.display_debug_info()

    # Todo: Handle background color
    """if background_bool:
        background = os.path.dirname(__file__) + "/inputs/" + "back " + graph_name.split(".pkl")[0] + ".ldr"
        background = read_bricks_from_file(background)
        selected_bricks += util.move_brickset(background, background_rgb, 0, 0, ldr_color)"""
    result_name = input("Please enter the image name: ")

    debugger = MyDebugger(f"{result_name}")
    write_bricks_to_file(
        selected_bricks,
        file_path=debugger.file_path(f"{result_name}.ldr"))

    print("done!")
