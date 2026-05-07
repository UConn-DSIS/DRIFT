import random
import time
import numpy as np
import torch
import torch as th
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.base import DGLError
import dgl.function as fn
import copy
import torch.nn.functional as F


class my_sampler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.t = 0

    def forward(self, train_ids, budget, reps):
        '''
            train_ids: all training ids in the current batch (list of indices)
            budget: budget_size (number of nodes to sample)
            reps: representations of all nodes, shape [n_nodes, n_features]
        '''
        n_nodes = len(train_ids)
        if n_nodes == 0:
            return []
        
        if budget >= n_nodes:
            return train_ids
        
        # reps_normalized = F.normalize(reps, p=2, dim=1)
        reps_normalized = F.softmax(reps, dim=1)
        
        # Initialize selected nodes
        ids_selected = []
        
        # First, select a random node or the node with maximum representation norm
        if len(ids_selected) == 0:
            # Select node with highest representation norm (before normalization)
            norms = torch.norm(reps, dim=1)
            first_idx = norms.argmax().item()
            ids_selected.append(first_idx)
        
        # Initialize minimum distances for diversity
        # min_dists[i] = minimum distance from node i to any selected node
        min_dists = torch.ones(n_nodes).to(reps.device) * 1000
        if len(ids_selected) > 0:
            min_dists = self.euclidean_similarity(reps_normalized[ids_selected[0]], reps_normalized)
            min_dists[ids_selected[0]] = -1000  # Mark as selected
        
        # Greedy diversity sampling: select nodes that maximize diversity
        # At each step, select the node with maximum minimum distance to already selected nodes
        for i in range(1, budget):
            if len(ids_selected) >= n_nodes:
                break
            
            # Find node with maximum minimum distance (most diverse from selected nodes)
            dist = min_dists.clone()
            dist[torch.tensor(ids_selected)] = -1000  # Exclude already selected
            
            if dist.max() < 0:
                # All nodes selected or invalid distances, break
                break
            
            next_idx = dist.argmax().item()
            ids_selected.append(next_idx)
            
            # Update minimum distances: for each node, keep track of its distance to the closest selected node
            new_dist = self.euclidean_similarity(reps_normalized[next_idx], reps_normalized)
            min_dists = torch.min(min_dists, new_dist)
            min_dists[next_idx] = -1000  # Mark as selected
        
        # Map back to original train_ids
        selected_train_ids = [train_ids[i] for i in ids_selected if i < len(train_ids)]
        
        return selected_train_ids

    
    def cosine_similarity(self, vec, mat):
        vec = vec.unsqueeze(0)
        return -torch.cosine_similarity(vec, mat, dim=1)/2+0.5
    
    def euclidean_similarity(self, vec, mat):
        vec = vec.unsqueeze(0)
        return torch.cdist(vec, mat, p=2).squeeze(0)
    
    def pairwise_euclidean_distance(self, x):
        # x should be a 2D tensor, shape: (batch_size, dim)
        # pairwise euclidean distance is calculated
        square = torch.sum(x**2, dim=1, keepdim=True)
        distance_square = -2 * torch.matmul(x, x.t()) + square + square.t()
        distance = torch.sqrt(distance_square + 1e-7) # add a small number to prevent numerical instability
        return distance

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rate):
        ctx.rate = rate
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input, rate):
        return GradReverse.apply(input, rate)
        
class DistingushModel(nn.Module):
    def __init__(self, IB_dim, dropout):
        super(DistingushModel, self).__init__()
        self.layer1 = GRL()
        self.layer2 = nn.Sequential(
            nn.Linear(IB_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            )

    def forward(self, h, rate=1.0):
        x1 = self.layer1(h, rate)
        output = self.layer2(x1)
        return output
    
    def reset_params(self):
        for layer in self.layer2:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()