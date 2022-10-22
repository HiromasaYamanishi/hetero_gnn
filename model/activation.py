import torch

class HeteroReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict):
        for k in x_dict.keys():
            x_dict[k] = x_dict[k].relu()
        return x_dict