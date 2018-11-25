from __future__ import absolute_import

import keras4hep as kh
from keras4hep.projects.qgjets.models import image

def build_a_model(model_type, model_name, *args, **kargs):
    models_subdir = getattr(kh.projects.qgjets.models, model_type)
    return getattr(models_subdir, model_name).build_a_model(*args, **kargs)

def get_custom_objects(model_type, model_name):
    models_subdir = getattr(kh.projects.qgjets.models, model_type)
    model_file = getattr(models_subdir, model_name)
    if hasattr(model_file, "get_custom_objects"):
        custom_objects = model_file.get_custom_objects()
    else:
        custom_objects = dict()
    return custom_objects
