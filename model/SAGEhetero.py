import torch
from torch_geometric.nn.dense.linear import Linear
import torch.nn as nn
from torch_scatter.scatter import scatter
from .activation import HeteroReLU

class SAGEHeteroConv(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, out_channels, pre_channels=None):
        super().__init__()
        num_nodes = {k: v[0] for k, v in x_dict.items()}

        if pre_channels== None:
            num_features = {k: v.size()[1] for k,v in x_dict.items()}
        else:
            num_features = {k: pre_channels for k in x_dict.keys()}

        self.linear = nn.ModuleDict({})
        for k in edge_index_dict.keys():
            self.linear['__'.join(k) + '__source'] = Linear(num_features[k[0]], out_channels, False, weight_initializer='glorot')
            self.linear['__'.join(k) + '__target'] = Linear(num_features[k[-1]], out_channels, False, weight_initializer='glorot')

    def forward(self, x_dict, edge_index_dict):
        x_dict_out = {}

        for k,v in edge_index_dict.items():
            source, target = k[0], k[-1]
            source_x = self.linear['__'.join(k) + '__source'](x_dict[source])
            target_x = self.linear['__'.join(k) + '__target'](x_dict[target])
            source_index = v[0].reshape(-1)
            target_index = v[1].reshape(-1)
            out = torch.zeros_like(target_x).to(target_x.device)
            source_x = source_x[source_index]

            target_x = target_x + scatter(source_x, target_index, out=out, dim=0, reduce='mean')
            if x_dict_out.get(target)!=None:
                x_dict_out[target] += target_x
            else:
                x_dict_out[target] = target_x

        #x_dict_out = {k: self.l2_norm(v) for k,v in x_dict_out.items()}    

        return x_dict_out

class MyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict):
        for k in x_dict.keys():
            x_dict[k] = x_dict[k].relu()
        return x_dict

        

class SAGEHetero(torch.nn.Module):
    def __init__(self, x_dict, edge_index_dict, hidden_channels, out_channels, target):
        super().__init__()
        self.conv1 = SAGEHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels=None)
        self.relu1 = HeteroReLU()
        self.conv2 = SAGEHeteroConv(x_dict, edge_index_dict, out_channels=hidden_channels, pre_channels= hidden_channels)
        self.relu2 = HeteroReLU()
        self.linear = Linear(hidden_channels, out_channels)

        self.target = target

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.relu1(x_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = self.relu2(x_dict)
        out = self.linear(x_dict[self.target])
        return out