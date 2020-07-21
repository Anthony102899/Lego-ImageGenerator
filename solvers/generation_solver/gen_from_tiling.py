from solvers.generation_solver.tile_graph import Tiling, generate_all_neighbor_tiles, unique_brick_list
import pickle5 as pickle
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
import time

def find_from_middle(tiling):
    debugger = tiling.middle_debugger
    tile_set = tiling.tile_set
    num_rings = tiling.num_rings
    result_tiles = tiling.result_tiles
    last_ring = tiling.last_ring  # the tiles in the last ring
    start_ring_idx = tiling.start_ring_idx
    last_idx = tiling.last_idx
    total_time = tiling.total_time

    for i in range(start_ring_idx, num_rings):
        print(f"\ncomputing ring {i}")
        if i == start_ring_idx:
            last_ring_num = tiling.last_ring_num
            start = last_idx + 1
        else:
            last_ring_num = len(last_ring)
            start = 0
        print("last_ring_num =  ", last_ring_num)
        start_time = time.time()
        t = time.time() - start_time + total_time
        # iterate over all bricks in the last ring
        for last_ring_idx in range(start, last_ring_num):
            print(f"last ring {last_ring_idx}")
            last_brick = last_ring.pop(0)  # brick instance in previous ring

            # brick instance to be aligned
            for align_tile in tile_set:
                # a list of neighbour bricks of "base_brick"
                neighbour_tiles = generate_all_neighbor_tiles(base_brick=last_brick, align_tile=align_tile, color=i + 1)
                neighbour_tiles = unique_brick_list(neighbour_tiles)

                for elem in neighbour_tiles:
                    if elem not in result_tiles:
                        result_tiles.append(elem)
                        last_ring.append(elem)
            t = time.time() - start_time + total_time
            if last_ring_idx % 2 == 1:
                tiling = Tiling(result_tiles, last_ring, last_ring_num,
                                debugger, tile_set, 
                                num_rings, i, last_ring_idx, 
                                t)
                pickle.dump(tiling, 
                            open(os.path.join(os.path.dirname(__file__), f'super_graph/r={i}#{num_rings}.pkl'), "wb"))
        write_bricks_to_file(
            result_tiles, file_path=debugger.file_path(f"n={len(result_tiles)} r={i+1} t={round(t, 2)}.ldr"))
    return result_tiles

if __name__ == "__main__": #os.path.join(os.path.dirname(__file__), f'super_graph/last={last_ring_idx} r={i}#{num_rings}.pkl')
    tiling = structure_graph = pickle.load(open(os.path.join(os.path.dirname(__file__), f'super_graph/r=1#4.pkl'), "rb"))
    num_rings = tiling.num_rings
    total_time = tiling.total_time
    start_time = time.time()

    bricks = find_from_middle(tiling)
    #debugger = tiling.middle_debugger
    #write_bricks_to_file(bricks, file_path=debugger.file_path(f"n={len(bricks)} r={num_rings} t={round(total_time + time.time() - start_time, 2)}.ldr"))