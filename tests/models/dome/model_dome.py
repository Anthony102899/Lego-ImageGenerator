from solvers.rigidity_solver.models import *
import numpy as np
from itertools import product

p = lambda x, y, z: np.array([x, y, z], dtype=np.double)
v = lambda x, y, z: np.array([x, y, z], dtype=np.double)
lerp = lambda p, q, w: p * (1 - w) + q * w

def polar_to_cart(r, phi, theta):
    return np.asarray([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(theta),
    ], dtype=np.double)

def define(stage):
    v_portions = 2
    h_portions = 2
    radius = 10
    v_max = np.pi / 2
    h_max = np.pi

    def beam_init(p, q):
        return Beam.tetra(p, q, thickness=2, density=0.3)

    ###### temp:
    theta = 0
    i = 0

    _p = {
        # axial points
        **{
            f"ax-far-l{i}-p{j}": polar_to_cart(radius, phi, theta)
            for j, phi in enumerate(np.linspace(0, h_max, h_portions + 1))
            # for i, theta in enumerate(np.linspace(0, v_max, v_portions, endpoint=False))
        },
        **{
            f"ax-near-l{i}-p{j}": polar_to_cart(radius * 0.9, phi, theta)
            for j, phi in enumerate(np.linspace(0, h_max, h_portions + 1))
            # for i, theta in enumerate(np.linspace(0, v_max, v_portions, endpoint=False))
        },
        **{
            f"internal-l{i}-p{j}": np.sum([f"ax-{n}-l{i}-p{k}" for n in ("near", "far") for k in (j, j + 1)]) / 4
            for j, phi in enumerate(np.linspace(h_max / h_portions / 2, h_max - h_max / h_portions / 2, h_portions))
            # for i, theta in enumerate(np.linspace(0, v_max, v_portions, endpoint=False))
        }
    # "top": polar_to_cart(radius, 0, 0.5 * np.pi),
    }
    _bmap = {
        **{
            f"ax-l{i}-p{j}": beam_init(_p[f"ax-far-l{i}-p{j}"], _p[f"ax-near-l{i}-p{j}"])
            for j in range(h_portions + 1)
            # for i in range(v_portions)
        },
        **{
            f"left-l{i}-j{j}": beam_init(_p[f"ax-far-l{i}-p{j}"], _p[f"ax-near-l{i}-p{j + 1}"])
            for j in range(h_portions)
            # for i in range(v_portions)
        },
        **{
            f"right-l{i}-p{j}": beam_init(_p[f"ax-far-l{i}-p{j}"], _p[f"ax-near-l{i}-p{j + 1}"])
            for j in range(h_portions)
            # for i in range(v_portions)
        },
    }
    joints = [
        *[],
    ]