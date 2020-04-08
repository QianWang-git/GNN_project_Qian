#!/usr/bin/env python
# coding: utf-8

# ## GNNStack notes
# **CS224W stanford**  
# **https://colab.research.google.com/drive/1DIQm9rOx2mT1bZETEeVUThxcrP1RKqAn#scrollTo=XyzIhe0O5ije**

# In[193]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from torchviz import make_dot
from torch.autograd import Variable
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch_geometric.datasets import CitationFull
from tqdm import tqdm
import pdb


data_cora = CitationFull('./CitationFull','cora')
data_cora_ml = CitationFull('./CitationFull','cora_ml')
data_citeseer =  CitationFull('./CitationFull','citeseer')
data_dblp =  CitationFull('./CitationFull','dblp')
data_pubmed =  CitationFull('./CitationFull','pubmed')



class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)    # symmetric normalized Laplacian
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    opt = optim.Adam(model.parameters(), lr=0.01)
    
    # train
    for epoch in range(150):
        total_loss = 0
        model.train()
        for batch in loader:
            #print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            #print(embedding.shape)     # why not batch size？？？？？？？？？？？？？？？？？？？ not 64
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        #if model.task == 'node':
            #mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            #pred = pred[mask]
            #label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            #print(len(data.y))
            #total += torch.sum(data.test_mask).item()
            total += len(data.y)
    #print(total)
    return correct / total


all_dataset =[data_cora,data_cora_ml,data_citeseer,data_dblp,data_pubmed]
all_dataset_name = ['cora','cora_ml','citeseer','dblp','pubmed']

for i,j in tqdm(zip(all_dataset,all_dataset_name)):
    writer = SummaryWriter("./log_"+j+'/' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    dataset = i.shuffle()
    task = 'node'
    model = train(dataset, task, writer)

# plot model structure
loader = DataLoader(dataset, batch_size=64, shuffle=True)
plot_model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
for batch in loader:
    y = plot_model(batch)
    g = make_dot(y, params=dict(plot_model.named_parameters()))
    g.view()
    break

