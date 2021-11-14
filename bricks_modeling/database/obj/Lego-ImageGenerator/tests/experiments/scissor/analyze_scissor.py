from model_scissor import define
import open3d as o3d

definition = define(1)

model = definition["model"]
print(definition["_p"])
model.visualize()
