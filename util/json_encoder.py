import json

import numpy as np
from solvers.rigidity_solver.models import Model, Joint, Beam

"""
    Helper package that implement classes to extend JSONEncoder
"""


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ModelEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Model):
            return {
                "beams": o.beams,
                "joints": o.joints,
            }
        elif isinstance(o, Beam):
            return {
                "principle_points": o.principle_points,
            }
        elif isinstance(o, Joint):
            return {
                "pivot": o.pivot,
                "rotation_dof": len(o.rotation_axes) if o.rotation_axes is not None else 0,
                "translation_dof": len(o.translation_vectors) if o.translation_vectors is not None else 0,
                "rotation_axes": o.rotation_axes,
                "translation_vectors": o.translation_vectors,
            }
        elif isinstance(o, np.ndarray):
            return o.tolist()
