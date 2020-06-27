from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.connections.conn_type import compute_conn_type
import itertools

""" Returns immediate possible aligns using "align_tile" for "base_brick" """
def get_all_tiles(base_brick: BrickInstance, align_tile: BrickInstance):
    result_tiles = []
    align_tags = []

    base_cpoints = base_brick.get_current_conn_points()  # a list of cpoints in base
    base_cpoint_num = len(base_cpoints)
    align_cpoints = align_tile.get_current_conn_points()  # a list of cpoints in align
    align_cpoint_num = len(align_cpoints)

    ls = [(x,y) for x in range(base_cpoint_num) for y in range(align_cpoint_num)]
    for base_cpoint_idx,align_cpoint_idx in ls:
        print("base cpoint num: ",base_cpoint_idx,"  align cpoint num: ",align_cpoint_idx)
        cpoint_base = base_cpoints[base_cpoint_idx]
        cpoint_align = base_cpoints[align_cpoint_idx]
        
        """
        1. determine whether they can connect
            2. get transformation
            3. result_tiles.append(new_tile)
        4. else continue
        """        
    return result_tiles

""" Return a list of "num_rings" neighbours of brick "base_brick" """
def form_complete_graph(num_rings: int, base_tile: BrickInstance, tile_set: list):
    result_tiles = [base_tile]  # the resulting tiles
    last_ring = [base_tile]  # the tiles in the last ring
    for i in range(0, num_rings):
        print(f"computing ring_{i}")
        last_ring_num = len(last_ring)
        for last_ring_idx in range(last_ring_num):  # iterate over all bricks in the last ring
            print(f"last ring_{last_ring_idx}")
            last_layer_brick = last_ring.pop(0)  # brick instance in previous ring
            for align_tile in tile_set:  # brick instance to be aligned
                neighbour_tiles = get_all_tiles(base_brick=last_layer_brick, align_tile=align_tile)  # a list of neighbour bricks of "base_brick"

                for elem in neighbour_tiles:
                    if elem not in result_tiles:
                        result_tiles.append(elem)
                        last_ring.append(elem)
    return result_tiles
