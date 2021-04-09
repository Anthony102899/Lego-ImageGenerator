import open3d as o3d

filenames = [
    "asdfasdf__sf.obj",
    "a1_.msh",
    "b1_.msh",
    "c1_.msh",
]

for fn in filenames:
    mesh = o3d.io.read_triangle_mesh(fn)
    mesh.paint_uniform_color([1, 0.706, 1])
    try:
        o3d.visualization.draw_geometries([mesh])
    except RuntimeError:
        pass
