# Custom model
import torch.nn as nn


class MergeModel(nn.Module):
    def __init__(self, model):
        super(MergeModel, self).__init__()

    def forward(self, x):
        pass
