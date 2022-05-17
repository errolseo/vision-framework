
def build_model(config):

    if config.type == "transformers":
        from .transformers import Transformers
        model = Transformers(config)

    elif config.type == "timm":
        from timm.models import create_model
        from .MetaFormer import MetaFG, MetaFG_meta

        model = create_model(config.name, **config.params)

    elif config.type == "torch":
        import torch
        raise Exception("To Do.")
        # _model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    else:
        raise Exception("Invalid model type.")

    return model



