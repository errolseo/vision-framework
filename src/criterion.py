
import torch.nn


def build_criterion(criterion):
    if not criterion:
        return None
    elif hasattr(torch.nn, criterion.name):
        _criterion = getattr(torch.nn, criterion.name)(**criterion.params)
    else:
        raise Exception("Not implemented.")

    return _criterion