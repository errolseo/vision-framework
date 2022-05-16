
from .merge import MergeModel
from .transforemrs import Transformers
from .MetaFormer.MetaFG import *

def build_model(model):

    if model.type == "transformers":
        _model = Transformers(model)
    elif model.type == "timm":
        import timm
        _model = timm.create_model(model.name, **model.params)
    elif model.type == "merge":
        _model = MergeModel(model)
    elif model.type == "torch":
        import torch
        raise Exception("To Do.")
        # _model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    else:
        raise Exception("Invalid model type.")

    return _model