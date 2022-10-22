import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import DBLP
import torch.optim as optim
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import HGTLoader
import os
import tqdm
import sys
sys.path.append('..')
from model.SAGEhetero import SAGEHetero
from model.SAGEAttention import SAGEAttention
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
dataset = DBLP(path)
data = dataset[0]
print(data)

# We initialize conference node features with a single feature.
data['conference'].x = torch.ones(data['conference'].num_nodes, 1)
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
print(device)
data= data.to(device)


train_loader = HGTLoader(
        data,
        num_samples={key: [512] * 4 for key in data.node_types},
        batch_size=128,
        input_nodes=('author', data['author'].train_mask),
    )

val_loader = HGTLoader(
        data,
        num_samples={key: [512] * 4 for key in data.node_types},
        batch_size=128,
        input_nodes=('author', data['author'].val_mask),
    )

test_loader = HGTLoader(
        data,
        num_samples={key: [512] * 4 for key in data.node_types},
        batch_size=128,
        input_nodes=('author', data['author'].test_mask),
    )


@torch.no_grad()
def init_params(train_loader, model):
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_index')
    model(batch.x_dict, batch.edge_index_dict)

def train(model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


def train_loader(loader, model, optimizer):
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        batch_size = batch['author'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['author'][:batch_size]
        loss = F.cross_entropy(out, batch['author'].y[:batch_size])
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss/ total_examples

@torch.no_grad()
def test_loader(loader, model):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device,  'edge_index')
        batch_size = batch['author'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['author'][:batch_size]
        pred = out.argmax(dim=1)

        total_examples += batch_size
        total_correct += int((pred==batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples

'''
for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
'''

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SAGEHetero')
    
    model_dict = {"SAGEHetero": SAGEHetero,
                "SAGEAttention": SAGEAttention}

    args = parser.parse_args()

    model = model_dict[args.model_name](data.x_dict, data.edge_index_dict, hidden_channels=128, out_channels=4, target='author')
    
    model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    for epoch in range(1, 101):
        loss = train(model, optimizer)
        train_acc, val_acc, test_acc = test(model)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
            f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
