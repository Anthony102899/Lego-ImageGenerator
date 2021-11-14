#%%
import numpy as np
from sfepy.discrete.fem import Mesh, FEDomain, Field

mesh_path = 'C:/Users/lzy71/miniconda3/envs/lego/lib/site-packages/sfepy/meshes/2d/rectangle_tri.mesh'

#%%
mesh = Mesh.from_file(mesh_path)
domain = FEDomain('domain', mesh)

min_x, max_x = domain.get_mesh_bounding_box()[:, 0]
eps = 1e-8 * (max_x - min_x)
omega = domain.create_region('Omega', 'all')

# %%
from sfepy.discrete import (
    FieldVariable, Material, Integral, Function,
    Equation, Equations, Problem
    )

field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=2)
u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')


# %%
from sfepy.mechanics.matcoefs import stiffness_from_lame
m = Material('m', D=stiffness_from_lame(dim=2, lam=1.0, mu=1.0))
f = Material('f', val=[[0.02], [0.01]])

integral = Integral('i', order=3)

# %%
from sfepy.terms import Term

t1 = Term.new('dw_lin_elastic(m.D, v, u)', 
    integral, omega, m=m, v=v, u=u)

t2 = Term.new('dw_volume_lvf(f.val, v)', 
    integral, omega, f=f, v=v)

eq = Equation('balance', t1 + t2)
eqs = Equations([eq])

#%%
pb = Problem('elasticity', equations=eqs)