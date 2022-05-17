
def build_optimizer(optimizer, model_params):

    if optimizer.type == "torch":
        import torch.optim
        _optimizer = getattr(torch.optim, optimizer.name)(model_params, **optimizer.params)
        
    else:
        raise Exception("Invalid optimizer type.")

    if "sam" in optimizer and optimizer.sam == True:
        from .sam import SAM
        _optimizer = SAM(_optimizer, optimizer)

    return _optimizer
