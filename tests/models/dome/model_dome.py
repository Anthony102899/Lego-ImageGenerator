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
    v_portions = 4
    h_portions = 8
    radius = 10
    _p = {
        **{
            f"l{i}-p{j}": polar_to_cart(radius, phi, theta)
            for j, phi in enumerate(np.linspace(0, 2 * np.pi, h_portions))
            for i, theta in enumerate(np.linspace(0, 0.5 * np.pi, v_portions, endpoint=False))
        },
        "top": polar_to_cart(radius, 0, 0.5 * np.pi),
    }
    _bmap = [
        *[
            Beam.tetra(_p[f"l{i}-p{j}"], _p[f"l{i}-p{(j + 1) % h_portions}"])
            for i in range(v_portions) for j in range(h_portions)
        ],
        *[
            Beam.tetra(_p[f"l{i}-p{j}"], _p[f"l{(i + 1) % v_portions}-p{j}"])
            for i in range(v_portions) for j in range(h_portions)
        ],
        *[
            Beam.tetra(_p[f"l{i}-p{v_portions - 1}"], _p["top"])
            for i in range(h_portions)
        ]
    ]
    joints = [

    ]