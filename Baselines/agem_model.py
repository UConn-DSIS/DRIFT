import pickle
import gc
import random
import torch
import torch.nn as nn
import torch.optim as optim
import copy

import numpy as np
#import quadprog
import dgl
from .agem_utils import store_grad, overwrite_grad

class NET(nn.Module):
    """
    A-GEM baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """
    def __init__(self,
                model,
                args,
                dataset=None):
        super(NET, self).__init__()        
        self.net = model

        self.ce = torch.nn.functional.cross_entropy
        self.opt = optim.Adam(self.net.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        
        # Store dataset for accessing full graph
        self.dataset = dataset
        if dataset is not None:
            self.full_graph = dataset.graph  # Full graph from dataset
        else:
            self.full_graph = None
        
        self.budget = int(args.agem_args['budget'][0])
        self.memory_proportion = int(args.agem_args['memory_proportion'][0])
        self.buffer_node_ids = []  # Store original node IDs from full graph
        self.aux_g = None
        self.aux_features = None
        self.aux_labels = None
        self.n_seen_examples = 0
        self.epochs = 0
        self.gpu = args.gpu
        self.cuda = args.cuda

        # allocate episodic memory
        # for semi-supervised data, it will store the training mask for every old tasks
        self.memory_data = []
        
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        if args.cuda:
            self.grads = torch.Tensor(sum(self.grad_dims), 2).cuda(args.gpu)
        else:
            self.grads = torch.Tensor(sum(self.grad_dims), 2)
        
        # allocate counters
        self.mem_cnt = 0
        self.old_dataloaders = {}
        self.seen_classes = []
        self.offset1 = 0
        self.offset2 = 0
    
    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, train_ids):
        """
        The method for learning under the task free online setting.
        
        :param args: Same as the args in __init__().
        :param g: The graph of the current batch.
        :param features: Node features of the current batch.
        :param labels: Labels of the nodes in the current batch.
        :param train_ids: The indices of the nodes participating in the training.
        """
        self.net.train()
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        
        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        
        # Get original node IDs from subgraph (mapping from remapped IDs to original IDs)
        train_ids_list = train_ids.tolist() if isinstance(train_ids, torch.Tensor) else list(train_ids)
        if '_ID' in g.ndata:
            original_node_ids = g.ndata['_ID'][train_ids_list].cpu().tolist()
        else:
            # Fallback: if _ID not available, use remapped IDs (should not happen in normal case)
            original_node_ids = train_ids_list
        
        # compute gradient on previous tasks (from buffer)
        if buffer_size > 0:
            sampled_mask = torch.zeros(buffer_size, dtype=torch.bool)
            n_samples = min(args.batch_size * self.memory_proportion, buffer_size)
            sampled_mask[torch.randperm(buffer_size)[:n_samples]] = True
            
            # calculate auxiliary loss based on replay
            self.net.zero_grad()
            nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
            
            # Get indices in aux_g (buffer indices correspond to aux_g node indices)
            sampled_indices = torch.where(sampled_mask)[0]
            if args.cuda:
                sampled_indices = sampled_indices.to(device='cuda:{}'.format(args.gpu))
            
            _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, sampled_indices)
            aux_features = aux_blocks[0].srcdata['feat']
            aux_output, _ = self.net.forward_batch(aux_blocks, aux_features)
            if isinstance(aux_output, tuple):
                aux_output = aux_output[0]
            
            old_task_loss = self.ce(aux_output, self.aux_labels[sampled_mask])
            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims, 0)

        # compute the grad on the current minibatch
        self.net.zero_grad()
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))
        
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        loss = self.ce(output, output_labels)
        loss.backward()

        # check if gradient violates constraints
        if buffer_size > 0:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, 1)
            # compute dot product of self.grads[:, 0] and self.grads[:, 1]
            dotp = torch.mm(self.grads[:, 1].unsqueeze(0),
                            self.grads.index_select(1, torch.LongTensor([0]).to(self.grads.device))) 
            if dotp < 0:
                dotp_ref = torch.mm(self.grads[:, 0].unsqueeze(0),
                            self.grads.index_select(1, torch.LongTensor([0]).to(self.grads.device))) 
                if dotp_ref.abs() > 1e-8:  # Avoid division by zero
                    projected_grad = (self.grads[:, 1] - (dotp / dotp_ref) * self.grads[:, 0]).squeeze()
                    # copy gradients back
                    overwrite_grad(self.net.parameters, projected_grad,
                                   self.grad_dims)
        
        self.opt.step()

        if last_epoch == 0:
            # perform reservoir sampling with original node IDs
            place_left = max(0, self.budget - len(self.buffer_node_ids))
            if place_left:
                offset = min(place_left, n_nodes)
                self.buffer_node_ids.extend(original_node_ids[:offset])
                if offset < n_nodes:
                    for i in range(offset, n_nodes):
                        j = torch.randint(0, self.n_seen_examples + i, (1,))
                        if j < self.budget:
                            self.buffer_node_ids[j] = original_node_ids[i]
            else:
                for i in range(n_nodes):
                    j = torch.randint(0, self.n_seen_examples + i, (1,))
                    if j < self.budget:
                        self.buffer_node_ids[j] = original_node_ids[i]
            self.n_seen_examples += n_nodes
            
            # Update aux graph using full graph
            self._update_aux_graph(args)

    def observe_cis(self, args, g, features, labels, train_ids):
        """
        The method for learning under the class-IL setting.
        
        :param args: Same as the args in __init__().
        :param g: The graph of the current batch.
        :param features: Node features of the current batch.
        :param labels: Labels of the nodes in the current batch.
        :param train_ids: The indices of the nodes participating in the training.
        """
        self.net.train()
        self.epochs += 1
        last_epoch = self.epochs % args.epochs
        
        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        
        # Get original node IDs from subgraph (mapping from remapped IDs to original IDs)
        train_ids_list = train_ids.tolist() if isinstance(train_ids, torch.Tensor) else list(train_ids)
        if '_ID' in g.ndata:
            original_node_ids = g.ndata['_ID'][train_ids_list].cpu().tolist()
        else:
            # Fallback: if _ID not available, use remapped IDs (should not happen in normal case)
            original_node_ids = train_ids_list
        
        # compute gradient on previous tasks (from buffer)
        if buffer_size > 0:
            sampled_mask = torch.zeros(buffer_size, dtype=torch.bool)
            n_samples = min(args.batch_size * self.memory_proportion, buffer_size)
            sampled_mask[torch.randperm(buffer_size)[:n_samples]] = True
            
            # calculate auxiliary loss based on replay
            self.net.zero_grad()
            nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
            
            # Get indices in aux_g (buffer indices correspond to aux_g node indices)
            sampled_indices = torch.where(sampled_mask)[0]
            if args.cuda:
                sampled_indices = sampled_indices.to(device='cuda:{}'.format(args.gpu))
            
            _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, sampled_indices)
            aux_features = aux_blocks[0].srcdata['feat']
            aux_output, _ = self.net.forward_batch(aux_blocks, aux_features)
            if isinstance(aux_output, tuple):
                aux_output = aux_output[0]
            
            # print(self.offset1, self.offset2)
            # print(f'{self.aux_labels.shape} labels range: {self.aux_labels[sampled_mask]}')
            # print(f'output range: {aux_output[:, self.offset1:self.offset2]}')
            old_task_loss = self.ce(aux_output[:, self.offset1:self.offset2], self.aux_labels[sampled_mask])
            old_task_loss.backward()
            store_grad(self.net.parameters, self.grads, self.grad_dims, 0)

        # compute the grad on the current minibatch
        self.net.zero_grad()
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        for label in labels[train_ids].unique():
            if label not in self.seen_classes:
                self.seen_classes.append(label)
        self.offset2 = max(self.seen_classes)+1
        if self.offset2 % 2 != 0:
            self.offset2 += 1
        
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        loss = self.ce(output[:, self.offset1:self.offset2], output_labels)
        loss.backward()

        # check if gradient violates constraints
        if buffer_size > 0:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, 1)
            # compute dot product of self.grads[:, 0] and self.grads[:, 1]
            dotp = torch.mm(self.grads[:, 1].unsqueeze(0),
                            self.grads.index_select(1, torch.LongTensor([0]).to(self.grads.device))) 
            if dotp < 0:
                dotp_ref = torch.mm(self.grads[:, 0].unsqueeze(0),
                            self.grads.index_select(1, torch.LongTensor([0]).to(self.grads.device))) 
                if dotp_ref.abs() > 1e-8:  # Avoid division by zero
                    projected_grad = (self.grads[:, 1] - (dotp / dotp_ref) * self.grads[:, 0]).squeeze()
                    # copy gradients back
                    overwrite_grad(self.net.parameters, projected_grad,
                                   self.grad_dims)
        
        self.opt.step()

        if last_epoch == 0:
            if args.setting == 'tfo_gaussian':
                place_left = max(0, self.budget - len(self.buffer_node_ids))
                if place_left:
                    offset = min(place_left, n_nodes)
                    self.buffer_node_ids.extend(train_ids[:offset])
                    if offset < n_nodes:
                        for i in range(offset, n_nodes):
                            j = torch.randint(0, self.n_seen_examples + i, (1,))
                            if j < self.budget:
                                self.buffer_node_ids[j] = train_ids[i]
                else:
                    for i in range(n_nodes):
                        j = torch.randint(0, self.n_seen_examples + i, (1,))
                        if j < self.budget:
                            self.buffer_node_ids[j] = train_ids[i]
                self.n_seen_examples += n_nodes
                subg = dgl.node_subgraph(g, self.buffer_node_ids, store_ids=True)
                n_edges = subg.edges()[0].shape[0]
                subg.remove_edges(list(range(n_edges)))
                subg = dgl.add_self_loop(subg)
                self.aux_g = subg.to(device='cuda:{}'.format(self.gpu))
                self.aux_features = self.aux_g.srcdata['feat']
                self.aux_labels = self.aux_g.dstdata['label'].squeeze()
            else:
                # perform reservoir sampling with original node IDs
                place_left = max(0, self.budget - len(self.buffer_node_ids))
                if place_left:
                    offset = min(place_left, n_nodes)
                    self.buffer_node_ids.extend(original_node_ids[:offset])
                    if offset < n_nodes:
                        for i in range(offset, n_nodes):
                            j = torch.randint(0, self.n_seen_examples + i, (1,))
                            if j < self.budget:
                                self.buffer_node_ids[j] = original_node_ids[i]
                else:
                    for i in range(n_nodes):
                        j = torch.randint(0, self.n_seen_examples + i, (1,))
                        if j < self.budget:
                            self.buffer_node_ids[j] = original_node_ids[i]
                self.n_seen_examples += n_nodes
                
                # Update aux graph using full graph
                self._update_aux_graph(args, labels)

    def _update_aux_graph(self, args, labels):
        """Update the auxiliary graph for buffer samples using the full graph"""
        if len(self.buffer_node_ids) == 0:
            return
        
        if self.full_graph is None:
            print("Warning: Full graph not available, cannot update aux graph")
            return
        
        # Create subgraph with buffer nodes from the full graph
        # buffer_node_ids stores original node IDs in the full graph
        buffer_node_ids_tensor = torch.tensor(self.buffer_node_ids, dtype=torch.long)
        if self.cuda:
            buffer_node_ids_tensor = buffer_node_ids_tensor.to(device='cuda:{}'.format(self.gpu))
        else:
            buffer_node_ids_tensor = buffer_node_ids_tensor.cpu()
        
        try:
            # Create subgraph from full graph using original node IDs
            if self.cuda:
                full_graph_gpu = self.full_graph.to(device='cuda:{}'.format(self.gpu))
            else:
                full_graph_gpu = self.full_graph
            
            subg = dgl.node_subgraph(full_graph_gpu, buffer_node_ids_tensor, store_ids=True)
            n_edges = subg.edges()[0].shape[0]
            subg.remove_edges(list(range(n_edges)))
            subg = dgl.add_self_loop(subg)
            
            if self.cuda:
                self.aux_g = subg.to(device='cuda:{}'.format(self.gpu))
            else:
                self.aux_g = subg
            
            self.aux_features = self.aux_g.srcdata['feat']
            self.aux_labels = self.aux_g.dstdata['label'].squeeze()
        except Exception as e:
            # If subgraph creation fails, skip update
            print(f"Warning: Could not update aux graph: {e}")
            pass
