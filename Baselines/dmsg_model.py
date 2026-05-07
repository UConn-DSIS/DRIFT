# compared to subgraph replay, this module removes the inter-task edges when loading stored subgraphs
# Adapted for TFOCGL (Task-Free Online Continual Graph Learning) setting
import torch
import copy
import dgl
from .dmsg_utils import *
from tqdm import tqdm
import math
import random

# samplers = {'My': my_sampler}
class NET(torch.nn.Module):
    def __init__(self,
                 model,
                 args,
                 dataset=None):
        super(NET, self).__init__()

        # setup network
        self.net = model

        self.distingush_model = DistingushModel(int(args.GCN_args['h_dims'][-1]), args.GCN_args['dropout'])
        if args.cuda:
            self.distingush_model.to(device='cuda:{}'.format(args.gpu))
        self.sampler = my_sampler(args)
        # self.sampler = samplers[args.sgreplay_args['sampler']](args)

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories for TFOCGL
        self.buffer_node_ids = []  # Original node IDs for reservoir sampling
        
        self.budget = int(args.sgreplay_args['budget'][0])
    
        
        # Handle loss_weights
        self.loss_weights = args.sgreplay_args.get('loss_weights', [1.0, 20.0, 1.0, 1.0])
        if isinstance(self.loss_weights, list) and len(self.loss_weights) > 0 and isinstance(self.loss_weights[0], list):
            self.loss_weights = self.loss_weights[0]
        
        # TFOCGL specific variables
        self.seen_classes = []
        self.offset1 = 0
        self.offset2 = 0
        self.epochs = 0
        self.n_seen_examples = 0
        
        # Store dataset for accessing full graph (similar to ER model)
        self.dataset = dataset
        if dataset is not None:
            self.full_graph = dataset.graph
        else:
            self.full_graph = None
        
        # Auxiliary graph and labels for replay
        self.aux_g = None
        self.aux_features = None
        self.aux_labels = None
        
        self.gpu = args.gpu
        self.cuda = args.cuda

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

        # Get original node IDs from subgraph
        train_ids_list = train_ids.tolist() if isinstance(train_ids, torch.Tensor) else list(train_ids)
        if '_ID' in g.ndata:
            original_node_ids = g.ndata['_ID'][train_ids_list].cpu().tolist()
        else:
            original_node_ids = train_ids_list

        self.net.zero_grad()
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        # Extract blocks for current batch
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_features = blocks[0].srcdata['feat']
        output, e_list = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        loss = self.ce(output, output_labels)

        # Compute auxiliary loss from buffer with DMSG diversity losses
        if buffer_size > 0 and self.aux_g is not None:
            # Sample nodes from buffer
            n_samples = min(args.batch_size, buffer_size)
            sampled_indices = torch.randperm(buffer_size)[:n_samples]
            if args.cuda:
                sampled_indices = sampled_indices.to(device='cuda:{}'.format(args.gpu))

            # Calculate auxiliary loss based on replay
            _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, sampled_indices)
            aux_features = aux_blocks[0].srcdata['feat']
            
            # Try to use variantion=True to get mu and std for diversity losses
            # If model doesn't support variantion, fall back to regular forward
            try:
                aux_output, aux_e_list, mu, std = self.net.forward_batch(aux_blocks, aux_features, variantion=True)
                use_diversity_losses = True
            except TypeError:
                # Model doesn't support variantion parameter, use regular forward
                aux_output, aux_e_list = self.net.forward_batch(aux_blocks, aux_features)
                use_diversity_losses = False
                mu, std = None, None
            
            if isinstance(aux_output, tuple):
                aux_output = aux_output[0]
            
            loss_aux = self.ce(aux_output, self.aux_labels[sampled_indices])
            
            # VAE loss and adversarial loss for diversity (only if variantion is supported)
            if use_diversity_losses and mu is not None and std is not None:
                h = self.net.second_last_h if hasattr(self.net, 'second_last_h') else aux_output
                similarity = torch.mm(h, h.t())
                label_similarity = (self.aux_labels[sampled_indices].unsqueeze(1) == self.aux_labels[sampled_indices].unsqueeze(0)).float()
                eye_mask = 1 - torch.eye(similarity.shape[0], device=similarity.device)
                loss_vae = torch.nn.functional.binary_cross_entropy_with_logits(similarity, label_similarity, weight=eye_mask)
                loss_vae += -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))

                # Adversarial loss for diversity
                h_mu = torch.cat((h, mu), dim=0)
                pre_stoch_true = self.distingush_model(h_mu).squeeze(-1)
                label_stoch_true = torch.cat((torch.ones(h.shape[0]), torch.zeros(mu.shape[0])), dim=0).to(h.device)
                loss_adv = torch.nn.functional.binary_cross_entropy_with_logits(pre_stoch_true, label_stoch_true)
            else:
                loss_vae = torch.tensor(0.0, device=loss.device)
                loss_adv = torch.tensor(0.0, device=loss.device)

            # Combine losses
            loss = self.loss_weights[0] * loss + self.loss_weights[1] * loss_aux + self.loss_weights[2] * loss_adv + self.loss_weights[3] * loss_vae
        else:
            # Only main loss if no buffer
            loss = self.loss_weights[0] * loss

        loss.backward()
        self.opt.step()

        # Update buffer: sample batch_size/2 nodes each time
        # If buffer is full, use reservoir sampling to replace
        self._update_buffer_batch(original_node_ids, g, args, train_ids_list, labels, blocks)

    def observe_cis(self, args, g, features, labels, train_ids):
        """
        The method for learning under the task free online class-incremental setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current batch.
        :param features: Node features of the current batch.
        :param labels: Labels of the nodes in the current batch.
        :param train_ids: The indices of the nodes participating in the training.
        """
        self.net.train()
        self.epochs += 1
        last_epoch = self.epochs % args.epochs

        # Update seen classes and output mask
        for label in labels[train_ids].unique():
            if label.item() not in self.seen_classes:
                self.seen_classes.append(label.item())
        self.offset2 = max(self.seen_classes) + 1
        if self.offset2 % 2 != 0:
            self.offset2 += 1

        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)

        # Get original node IDs from subgraph
        train_ids_list = train_ids.tolist() if isinstance(train_ids, torch.Tensor) else list(train_ids)

        if '_ID' in g.ndata:
            original_node_ids = g.ndata['_ID'][train_ids_list].cpu().tolist()
        else:
            original_node_ids = train_ids_list

        self.net.zero_grad()
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        # Extract blocks for current batch
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_features = blocks[0].srcdata['feat']
        output, e_list = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        loss = self.ce(output[:, self.offset1:self.offset2], output_labels)

        # Compute auxiliary loss from buffer with DMSG diversity losses
        if buffer_size > 0 and self.aux_g is not None:
            # Sample nodes from buffer
            n_samples = min(args.batch_size, buffer_size)
            sampled_indices = torch.randperm(buffer_size)[:n_samples]
            if args.cuda:
                sampled_indices = sampled_indices.to(device='cuda:{}'.format(args.gpu))

            # Calculate auxiliary loss based on replay
            _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, sampled_indices)
            aux_features = aux_blocks[0].srcdata['feat']
            
            aux_output, _, mu, std = self.net.forward_batch(aux_blocks, aux_features, variantion=True)
            
            if isinstance(aux_output, tuple):
                aux_output = aux_output[0]
            
            loss_aux = self.ce(aux_output[:, self.offset1:self.offset2], self.aux_labels[sampled_indices])
            
            # VAE loss and adversarial loss for diversity (only if variantion is supported)
            if mu is not None and std is not None:
                h = self.net.second_last_h if hasattr(self.net, 'second_last_h') else aux_output
                similarity = torch.mm(h, h.t())
                label_similarity = (self.aux_labels[sampled_indices].unsqueeze(1) == self.aux_labels[sampled_indices].unsqueeze(0)).float()
                eye_mask = 1 - torch.eye(similarity.shape[0], device=similarity.device)
                loss_vae = torch.nn.functional.binary_cross_entropy_with_logits(similarity, label_similarity, weight=eye_mask)
                loss_vae += -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))

                # Adversarial loss for diversity
                h_mu = torch.cat((h, mu), dim=0)
                pre_stoch_true = self.distingush_model(h_mu).squeeze(-1)
                label_stoch_true = torch.cat((torch.ones(h.shape[0]), torch.zeros(mu.shape[0])), dim=0).to(h.device)
                loss_adv = torch.nn.functional.binary_cross_entropy_with_logits(pre_stoch_true, label_stoch_true)
            else:
                loss_vae = torch.tensor(0.0, device=loss.device)
                loss_adv = torch.tensor(0.0, device=loss.device)

            # Combine losses
            loss = self.loss_weights[0] * loss + self.loss_weights[1] * loss_aux + self.loss_weights[2] * loss_adv + self.loss_weights[3] * loss_vae

        loss.backward()
        self.opt.step()

        # If buffer is full, use reservoir sampling to replace
        if args.setting == 'tfo_gaussian':
            self._update_buffer_batch_local(g, args, train_ids_list, labels, blocks)
        else:  
            self._update_buffer_batch(original_node_ids, g, args, train_ids_list, labels, blocks)

    def _update_buffer_batch(self, original_node_ids, g, args, train_ids_list, labels, blocks):
        """
        Update buffer: use DMSG sampler to sample batch_size/2 nodes from current batch.
        If buffer is not full, add directly. If full, use reservoir sampling to replace.
        """
        n_nodes = len(original_node_ids)
        sample_size = max(1, args.batch_size // 2)  # Sample batch_size/2 nodes
        sample_size = min(sample_size, n_nodes)  # Don't exceed available nodes
        
        if n_nodes == 0:
            return
        
        # Get representations for current batch nodes
        self.net.eval()
        with torch.no_grad():
            input_features = blocks[0].srcdata['feat']
            output, _ = self.net.forward_batch(blocks, input_features)
            if isinstance(output, tuple):
                output = output[0]
            # Use output as representations
            current_reps = output.cpu()
        self.net.train()
        
        # Prepare train_ids as list of indices [0, 1, ..., n_nodes-1]
        train_ids_indices = list(range(n_nodes))
        
        # Use DMSG sampler to select nodes from current batch
        # The sampler now works directly on all train_ids without class-wise grouping
        if len(train_ids_indices) > 0 and len(current_reps) > 0:
            try:
                # The sampler expects:
                # - train_ids: list of indices [0, 1, ..., n_nodes-1]
                # - budget: number of nodes to sample
                # - reps: representations tensor [n_nodes, n_features]
                selected_indices = self.sampler(train_ids_indices, sample_size, current_reps)
                # selected_indices are indices in train_ids_list, map to original_node_ids
                sampled_node_ids = [original_node_ids[i] for i in selected_indices if i < len(original_node_ids)]
            except Exception as e:
                print(f"Warning: Sampler failed, using random sampling: {e}")
                # Fall back to random sampling
                if n_nodes > sample_size:
                    sampled_indices = torch.randperm(n_nodes)[:sample_size]
                    sampled_node_ids = [original_node_ids[i] for i in sampled_indices]
                else:
                    sampled_node_ids = original_node_ids
        else:
            # Fall back to random sampling if no valid nodes or representations
            if n_nodes > sample_size:
                sampled_indices = torch.randperm(n_nodes)[:sample_size]
                sampled_node_ids = [original_node_ids[i] for i in sampled_indices]
            else:
                sampled_node_ids = original_node_ids
        
        buffer_size = len(self.buffer_node_ids)
        place_left = self.budget - buffer_size
        
        # If buffer has space, add directly
        if place_left > 0:
            add_count = min(place_left, len(sampled_node_ids))
            self.buffer_node_ids.extend(sampled_node_ids[:add_count])
            self.n_seen_examples += add_count
            
            # If there are remaining nodes and buffer is now full, use reservoir sampling
            if add_count < len(sampled_node_ids) and len(self.buffer_node_ids) >= self.budget:
                remaining_nodes = sampled_node_ids[add_count:]
                for node_id in remaining_nodes:
                    j = torch.randint(0, self.n_seen_examples, (1,))
                    if j < self.budget:
                        self.buffer_node_ids[j] = node_id
                    self.n_seen_examples += 1
        else:
            # Buffer is full, use reservoir sampling to replace
            for node_id in sampled_node_ids:
                j = torch.randint(0, self.n_seen_examples, (1,))
                if j < self.budget:
                    self.buffer_node_ids[j] = node_id
                self.n_seen_examples += 1
        
        # Update aux graph using full graph (similar to ER model)
        if len(self.buffer_node_ids) == 0:
            return
        
        if self.full_graph is None:
            print("Warning: Full graph not available, cannot update aux graph")
            return
        
        try:
            # Create subgraph with buffer nodes from the full graph
            buffer_node_ids_tensor = torch.tensor(self.buffer_node_ids, dtype=torch.long)
            if self.cuda:
                buffer_node_ids_tensor = buffer_node_ids_tensor.to(device='cuda:{}'.format(self.gpu))
            else:
                buffer_node_ids_tensor = buffer_node_ids_tensor.cpu()
            
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
            print(f"Warning: Could not update aux graph: {e}")
            pass

    def _update_buffer_batch_local(self, g, args, train_ids_list, labels, blocks):
        """
        Update buffer: use DMSG sampler to sample batch_size/2 nodes from current batch.
        If buffer is not full, add directly. If full, use reservoir sampling to replace.
        """
        n_nodes = len(train_ids_list)
        sample_size = max(1, args.batch_size // 2)  # Sample batch_size/2 nodes
        sample_size = min(sample_size, n_nodes)  # Don't exceed available nodes
        
        if n_nodes == 0:
            return
        
        # Get representations for current batch nodes
        self.net.eval()
        with torch.no_grad():
            input_features = blocks[0].srcdata['feat']
            output, _ = self.net.forward_batch(blocks, input_features)
            if isinstance(output, tuple):
                output = output[0]
            # Use output as representations
            current_reps = output.cpu()
        self.net.train()
        
        # Prepare train_ids as list of indices [0, 1, ..., n_nodes-1]
        train_ids_indices = list(range(n_nodes))
        
        # Use DMSG sampler to select nodes from current batch
        # The sampler now works directly on all train_ids without class-wise grouping
        if len(train_ids_indices) > 0 and len(current_reps) > 0:
            try:
                # The sampler expects:
                # - train_ids: list of indices [0, 1, ..., n_nodes-1]
                # - budget: number of nodes to sample
                # - reps: representations tensor [n_nodes, n_features]
                selected_indices = self.sampler(train_ids_indices, sample_size, current_reps)
                # selected_indices are indices in train_ids_list, map to original_node_ids
                sampled_node_ids = [train_ids_list[i] for i in selected_indices if i < len(train_ids_list)]
            except Exception as e:
                print(f"Warning: Sampler failed, using random sampling: {e}")
                # Fall back to random sampling
                if n_nodes > sample_size:
                    sampled_indices = torch.randperm(n_nodes)[:sample_size]
                    sampled_node_ids = [train_ids_list[i] for i in sampled_indices]
                else:
                    sampled_node_ids = train_ids_list
        else:
            # Fall back to random sampling if no valid nodes or representations
            if n_nodes > sample_size:
                sampled_indices = torch.randperm(n_nodes)[:sample_size]
                sampled_node_ids = [train_ids_list[i] for i in sampled_indices]
            else:
                sampled_node_ids = train_ids_list
        
        buffer_size = len(self.buffer_node_ids)
        place_left = self.budget - buffer_size
        
        # If buffer has space, add directly
        if place_left > 0:
            add_count = min(place_left, len(sampled_node_ids))
            self.buffer_node_ids.extend(sampled_node_ids[:add_count])
            self.n_seen_examples += add_count
            
            # If there are remaining nodes and buffer is now full, use reservoir sampling
            if add_count < len(sampled_node_ids) and len(self.buffer_node_ids) >= self.budget:
                remaining_nodes = sampled_node_ids[add_count:]
                for node_id in remaining_nodes:
                    j = torch.randint(0, self.n_seen_examples, (1,))
                    if j < self.budget:
                        self.buffer_node_ids[j] = node_id
                    self.n_seen_examples += 1
        else:
            # Buffer is full, use reservoir sampling to replace
            for node_id in sampled_node_ids:
                j = torch.randint(0, self.n_seen_examples, (1,))
                if j < self.budget:
                    self.buffer_node_ids[j] = node_id
                self.n_seen_examples += 1
        
        # Update aux graph using full graph (similar to ER model)
        if len(self.buffer_node_ids) == 0:
            return
        
        if self.full_graph is None:
            print("Warning: Full graph not available, cannot update aux graph")
            return
        
        try:
            subg = dgl.node_subgraph(g, self.buffer_node_ids, store_ids=True)
            n_edges = subg.edges()[0].shape[0]
            subg.remove_edges(list(range(n_edges)))
            subg = dgl.add_self_loop(subg)
            self.aux_g = subg.to(device='cuda:{}'.format(self.gpu))
            self.aux_features = self.aux_g.srcdata['feat']
            self.aux_labels = self.aux_g.dstdata['label'].squeeze()
            
        except Exception as e:
            print(f"Warning: Could not update aux graph: {e}")
            pass