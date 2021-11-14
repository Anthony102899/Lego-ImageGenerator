"""
Compute the stiffness matrix from a mesh file
"""
import six

import numpy as np

from sfepy.discrete.fem import Mesh
from sfepy.base.conf import ProblemConf, get_standard_keywords
from sfepy.discrete import Problem


def define(mesh_path):
    from sfepy.mechanics.matcoefs import stiffness_from_lame

    filename_mesh = Mesh.from_file(mesh_path)

    options = {
        'nls': 'newton',
        'ls': 'ls',
        'ts': 'ts',
        'save_times': 'all',
    }

    functions = {
        'linear_tension': (linear_tension,),
        'empty': (lambda ts, coor, mode, region, ig: None,),
    }

    fields = {
        'displacement': ('real', 3, 'Omega', 1),
    }

    # Coefficients are chosen so that the tangent stiffness is the same for all
    # material for zero strains.
    # Young modulus = 10 kPa, Poisson's ratio = 0.3
    materials = {
        'solid': ({
                      'K': 8.333,  # bulk modulus
                      'D': stiffness_from_lame(dim=3, lam=5.769, mu=3.846),
                  },),
        'load': 'empty',
    }

    variables = {
        'u': ('unknown field', 'displacement', 0),
        'v': ('test field', 'displacement', 'u'),
    }

    regions = {
        'Omega': 'all',
        # 'Bottom': ('vertices in (z < 0.1)', 'facet'),
        'Top': ('vertices in (z > -100)', 'facet'),
    }

    ebcs = { }

    integrals = {
        'i': 1,
        'isurf': 2,
    }

    equations = {
        'linear': """dw_lin_elastic.i.Omega(solid.D, v, u)
                    = dw_surface_ltr.isurf.Top(load.val, v)""",
    }

    solvers = {
        'ls': ('ls.scipy_direct', {}),
        'newton': ('nls.newton', {
            'i_max': 2,
            'eps_a': 1e-10,
            'eps_r': 1.0,
        }),
        'ts': ('ts.simple', {
            't0': 0,
            't1': 1,
            'dt': None,
            'n_step': 2,  # has precedence over dt!
            'verbose': 1,
        }),
    }

    return locals()


def linear_tension(ts, coor, mode=None, **kwargs):
    if mode == 'qp':
        val = np.tile(0.1 * ts.step, (coor.shape[0], 1, 1))
        return {'val': val}


def stiffness_matrix_from_mesh(mesh_path):
    required, other = get_standard_keywords()
    conf = ProblemConf.from_file(__file__, required, other, define_args=(mesh_path,))
    problem = Problem.from_conf(conf, init_equations=False)

    for key, eq in six.iteritems(problem.conf.equations):
        problem.set_equations({key: eq})
        load = problem.get_materials()['load']
        load.set_function(linear_tension)
        problem.solve(save_results=False, verbose=False)

    K = problem.mtx_a
    return K


if __name__ == '__main__':
    mesh_path = "./data/cube.mesh"
    stiffness_matrix_from_mesh(mesh_path)
