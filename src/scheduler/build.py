
def build_scheduler(optimizer, config):

    if config.type == "torch":
        import torch.optim.lr_scheduler
        scheduler = getattr(torch.optim.lr_scheduler, config.name)(optimizer, **config.params)

    elif config.type == "CosineAnnealingWarmUpRestarts":
        from .CosineAnnealingWarmUpRestarts import CosineAnnealingWarmUpRestarts
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, **config.params)

    else:
        raise Exception("Invalid scheduler type.")

    return scheduler
