import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dgl

# Auxiliary functions
def get_grad_vector(pp, grad_dims):
    """
    Gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def add_memory_grad(pp, mem_grads, grad_dims):
    """
    This stores the gradient of a new memory and compute the dot product with the previously stored memories.
    pp: parameters
    mem_grads: gradients of previous memories
    grad_dims: list with number of parameters per layers
    """
    # gather the gradient of the new memory
    grads = get_grad_vector(pp, grad_dims)

    if mem_grads is None:
        mem_grads = grads.unsqueeze(dim=0)
    else:
        grads = grads.unsqueeze(dim=0)
        mem_grads = torch.cat((mem_grads, grads), dim=0)

    return mem_grads


class NET(nn.Module):
    """
    GSS (Gradient-based Sample Selection) baseline for NCGL tasks
    
    Adapted from the original GSS implementation for graph neural networks.
    This method uses gradient-based similarity to select samples for the replay buffer.
    
    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting configurations like class-IL and task-IL, etc.
    """
    
    def __init__(self, model, args, dataset=None):
        super(NET, self).__init__()
        
        # backbone network
        self.net = model
        
        # setup optimizer
        self.opt = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # setup loss
        self.ce = torch.nn.functional.cross_entropy
        
        # Store dataset for accessing full graph
        self.dataset = dataset
        self.full_graph = dataset.graph  # Full graph from dataset
        
        # GSS-specific parameters
        self.rn = args.gss_args.get('memory_strength', 5)  # number of gradient vectors to estimate new samples similarity
        self.n_memories = args.gss_args.get('n_memories', args.batch_size)  # auxiliary storage before deciding samples to the buffer
        self.n_sampled_memories = args.gss_args.get('n_sampled_memories', 100)  # buffer size, M
        self.n_constraints = args.gss_args.get('n_constraints', args.batch_size)  # n_samples to be replayed from the buffer
        self.gpu = args.gpu
        self.cuda = args.cuda
        
        self.batch_size = args.batch_size
        self.n_iter = args.gss_args.get('n_iter', 1)  # number of iterations (update steps) for each received batch
        self.sim_th = args.gss_args.get('change_th', 0.0)  # cosine similarity threshold for being a candidate for buffer entrance
        self.subselect = args.gss_args.get('subselect', args.batch_size)  # for estimating new samples score, draw samples in batch of size subselect
        
        # Allocate ring buffer (stores node IDs)
        self.memory_node_ids = []
        self.memory_labs = []
        self.mem_cnt = 0
        
        # Allocate buffer
        self.sampled_memory_node_ids = None
        self.sampled_memory_labs = None
        self.sampled_memory_cos = None  # buffer cosine similarity score
        
        # Allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        
        # Store graph and features for buffer samples
        self.aux_g = None
        self.aux_features = None
        self.aux_labels = None

        self.seen_classes = []
        self.offset1 = 0
        self.offset2 = 0

    def forward(self, features):
        output = self.net(features)
        return output

    def cosine_similarity(self, x1, x2=None, eps=1e-8):
        x2 = x1 if x2 is None else x2
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        # avoid division by zero
        w1 = w1 + eps
        w2 = w2 + eps
        sim = torch.mm(x1, x2.t()) / (w1 * w2.t())
        # sim[torch.isinf(sim) | torch.isnan(sim)] = 0.0

        # x2 = x1 if x2 is None else x2
        # w1 = x1.norm(p=2, dim=1, keepdim=True)

        # w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
        # sim= torch.mm(x1, x2.t())/(w1 * w2.t()) #, w1  # .clamp(min=eps), 1/cosinesim

        return sim

    def print_buffer_stats(self):
        """Print buffer statistics (number of samples in buffer)"""
        if self.sampled_memory_node_ids is None:
            print('Buffer is empty')
        else:
            print(f'Buffer size: {len(self.sampled_memory_node_ids)} samples')

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
        
        # Update ring buffer storing examples from current batch
        bsz = len(train_ids)
        train_ids_list = train_ids.tolist() if isinstance(train_ids, torch.Tensor) else list(train_ids)
        
        # Get original node IDs from subgraph (mapping from remapped IDs to original IDs)
        if '_ID' in g.ndata:
            original_node_ids = g.ndata['_ID'][train_ids_list].cpu().tolist()
        else:
            # Fallback: if _ID not available, use remapped IDs (should not happen in normal case)
            original_node_ids = train_ids_list
        
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        
        # Store original node IDs and labels
        self.memory_node_ids.extend(original_node_ids[:effbsz])
        labels_list = labels[train_ids].tolist() if isinstance(labels[train_ids], torch.Tensor) else labels[train_ids].cpu().numpy().tolist()
        self.memory_labs.extend(labels_list[:effbsz])
        self.mem_cnt += effbsz
        
        # Prepare for replay if buffer exists
        if self.sampled_memory_node_ids is not None and len(self.sampled_memory_node_ids) > 0:
            shuffled_inds = torch.randperm(len(self.sampled_memory_node_ids))
            effective_batch_size = min(self.n_constraints, len(self.sampled_memory_node_ids))
            b_index = 0
        
        # Gradients of used buffer samples
        self.mem_grads = None
        
        # Neighbor sampler
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))
        
        for iter_i in range(self.n_iter):
            # Compute grad on current minibatch and perform update step
            self.net.zero_grad()
            
            _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
            input_features = blocks[0].srcdata['feat']
            output, _ = self.net.forward_batch(blocks, input_features)
            output_labels = labels[train_ids]
            
            if isinstance(output, tuple):
                output = output[0]
            
            if self.offset2 == 0:
                self.offset2 = len(output[1])
            
            loss = self.ce(output, output_labels)
            loss.backward()
            this_grad = get_grad_vector(self.net.parameters, self.grad_dims).unsqueeze(0)
            self.opt.step()
            
            # Update steps on the replayed samples from buffer
            if self.sampled_memory_node_ids is not None and len(self.sampled_memory_node_ids) > 0:
                random_batch_inds = shuffled_inds[b_index * effective_batch_size: b_index * effective_batch_size + effective_batch_size]
                random_batch_inds = random_batch_inds.tolist()
                
                # random_batch_inds are indices in the buffer, which correspond to node indices in aux_g
                # (since aux_g is created from buffer nodes in order)
                batch_node_indices = torch.tensor(random_batch_inds, dtype=torch.long)
                if args.cuda:
                    batch_node_indices = batch_node_indices.to(device='cuda:{}'.format(args.gpu))
                
                # Sample blocks for buffer nodes
                _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, batch_node_indices)
                batch_x = aux_blocks[0].srcdata['feat']
                # Get labels from aux_labels using indices
                if isinstance(self.aux_labels, torch.Tensor):
                    batch_y = self.aux_labels[random_batch_inds]
                else:
                    batch_y = torch.tensor([self.aux_labels[i] for i in random_batch_inds], dtype=torch.long)
                if args.cuda:
                    batch_y = batch_y.to(device='cuda:{}'.format(args.gpu))
                
                self.net.zero_grad()
                output_aux, _ = self.net.forward_batch(aux_blocks, batch_x)
                if isinstance(output_aux, tuple):
                    output_aux = output_aux[0]
                loss_aux = self.ce(output_aux, batch_y)
                loss_aux.backward()
                self.opt.step()
                
                b_index += 1
                if b_index * effective_batch_size >= len(self.sampled_memory_node_ids):
                    b_index = 0
        
        # Memory update: when ring buffer is full
        if self.mem_cnt == self.n_memories:
            self.net.eval()
            
            if self.sampled_memory_node_ids is not None and self.n_sampled_memories <= len(self.sampled_memory_node_ids):
                # Buffer is full - decide on replacement
                effective_batch_size = min(self.n_constraints, len(self.sampled_memory_node_ids))
                batch_sim = self.get_batch_sim(args, g, features, labels, effective_batch_size)
                
                if batch_sim < self.sim_th:
                    # Prepare new samples
                    mem_node_ids = self.memory_node_ids.copy()
                    mem_labs = self.memory_labs.copy()
                    
                    # Normalize buffer similarity scores
                    # Handle inf, nan, and edge cases
                    cos_scores = self.sampled_memory_cos.clone()
                    # Replace inf and nan with 0
                    cos_scores[torch.isinf(cos_scores)] = 0.0
                    cos_scores[torch.isnan(cos_scores)] = 0.0
                    
                    cos_min = torch.min(cos_scores)
                    cos_max = torch.max(cos_scores)
                    cos_range = cos_max - cos_min
                    
                    if cos_range < 1e-6:
                        # All values are the same, use uniform distribution
                        buffer_sim = torch.ones_like(cos_scores) / len(cos_scores)
                    else:
                        # Normalize to [0, 1]
                        buffer_sim = (cos_scores - cos_min) / (cos_range + 1e-6)
                        # Ensure all values are non-negative and finite
                        buffer_sim = torch.clamp(buffer_sim, min=0.0, max=1.0)
                        # Replace any remaining inf/nan with 0
                        buffer_sim[torch.isinf(buffer_sim) | torch.isnan(buffer_sim)] = 0.0
                        # Normalize to sum to 1 for multinomial
                        buffer_sim = buffer_sim / (buffer_sim.sum() + 1e-8)
                    
                    # Draw candidates for replacement from the buffer
                    index = torch.multinomial(buffer_sim, len(mem_node_ids), replacement=False)
                    index = index.tolist()
                    
                    # Estimate similarity of each sample in received batch
                    batch_item_sim = self.get_each_batch_sample_sim(args, g, features, labels)
                    
                    # Scale similarity scores to [0, 1] for probability
                    batch_sim_scaled = batch_item_sim.clone()
                    batch_sim_scaled[torch.isinf(batch_sim_scaled) | torch.isnan(batch_sim_scaled)] = 0.0
                    scaled_batch_item_sim = torch.clamp((batch_sim_scaled + 1) / 2, min=0.0, max=1.0).unsqueeze(1).clone()
                    
                    buffer_cos_selected = self.sampled_memory_cos[index].clone()
                    buffer_cos_selected[torch.isinf(buffer_cos_selected) | torch.isnan(buffer_cos_selected)] = 0.0
                    buffer_repl_batch_sim = torch.clamp((buffer_cos_selected + 1) / 2, min=0.0, max=1.0).unsqueeze(1).clone()
                    
                    # Combine probabilities and normalize
                    combined_probs = torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1)
                    # Normalize each row to sum to 1
                    combined_probs = combined_probs / (combined_probs.sum(dim=1, keepdim=True) + 1e-8)
                    
                    # Draw an event to decide on replacement decision
                    outcome = torch.multinomial(combined_probs, 1, replacement=False)
                    
                    # Replace samples with outcome = 1
                    added_indx = torch.arange(end=batch_item_sim.size(0))
                    sub_index = outcome.squeeze(1).byte()
                    
                    for idx, replace in enumerate(sub_index):
                        if replace:
                            buffer_idx = index[idx]
                            self.sampled_memory_node_ids[buffer_idx] = mem_node_ids[idx]
                            self.sampled_memory_labs[buffer_idx] = mem_labs[idx]
                            self.sampled_memory_cos[buffer_idx] = batch_item_sim[idx].clone()
                    
                    # Update aux graph
                    self._update_aux_graph(g, features, labels)
            else:
                # Add new samples to the buffer
                added_inds = list(range(len(self.memory_node_ids)))
                
                # First buffer insertion
                if self.sampled_memory_node_ids is None:
                    self.sampled_memory_node_ids = self.memory_node_ids.copy()
                    self.sampled_memory_labs = self.memory_labs.copy()
                    self.sampled_memory_cos = torch.zeros(len(added_inds)) + 0.1
                else:
                    effective_batch_size = min(self.n_constraints, len(self.sampled_memory_node_ids))
                    self.get_batch_sim(args, g, features, labels, effective_batch_size)  # Draw random samples from buffer
                    this_sampled_memory_cos = self.get_each_batch_sample_sim(args, g, features, labels).clone()
                    
                    self.sampled_memory_cos = torch.cat((self.sampled_memory_cos, this_sampled_memory_cos.clone()), dim=0)
                    self.sampled_memory_node_ids.extend(self.memory_node_ids)
                    self.sampled_memory_labs.extend(self.memory_labs)
                
                # Update aux graph
                self._update_aux_graph(g, features, labels)
            
            # self.print_buffer_stats()
            self.mem_cnt = 0
            self.memory_node_ids = []
            self.memory_labs = []
            self.net.train()

    def observe_cis(self, args, g, features, labels, train_ids):
        """
        The method for learning under the task free online setting.
        
        :param args: Same as the args in __init__().
        :param g: The graph of the current batch.
        :param features: Node features of the current batch.
        :param labels: Labels of the nodes in the current batch.
        :param train_ids: The indices of the nodes participating in the training.
        """
        self.net.train()
        
        for label in labels[train_ids].unique():
            if label not in self.seen_classes:
                self.seen_classes.append(label)
        self.offset2 = max(self.seen_classes)+1
        if self.offset2 % 2 != 0:
            self.offset2 += 1

        # Update ring buffer storing examples from current batch
        bsz = len(train_ids)
        train_ids_list = train_ids.tolist() if isinstance(train_ids, torch.Tensor) else list(train_ids)
        
        # Get original node IDs from subgraph (mapping from remapped IDs to original IDs)
        if '_ID' in g.ndata:
            original_node_ids = g.ndata['_ID'][train_ids_list].cpu().tolist()
        else:
            # Fallback: if _ID not available, use remapped IDs (should not happen in normal case)
            original_node_ids = train_ids_list
        
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        
        # Store original node IDs and labels
        self.memory_node_ids.extend(original_node_ids[:effbsz])
        labels_list = labels[train_ids].tolist() if isinstance(labels[train_ids], torch.Tensor) else labels[train_ids].cpu().numpy().tolist()
        self.memory_labs.extend(labels_list[:effbsz])
        self.mem_cnt += effbsz
        
        # Prepare for replay if buffer exists
        if self.sampled_memory_node_ids is not None and len(self.sampled_memory_node_ids) > 0:
            shuffled_inds = torch.randperm(len(self.sampled_memory_node_ids))
            effective_batch_size = min(self.n_constraints, len(self.sampled_memory_node_ids))
            b_index = 0
        
        # Gradients of used buffer samples
        self.mem_grads = None
        
        # Neighbor sampler
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))
        
        for iter_i in range(self.n_iter):
            # Compute grad on current minibatch and perform update step
            self.net.zero_grad()
            
            _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
            input_features = blocks[0].srcdata['feat']
            output, _ = self.net.forward_batch(blocks, input_features)
            output_labels = labels[train_ids]
            
            if isinstance(output, tuple):
                output = output[0]
            
            loss = self.ce(output[:, self.offset1:self.offset2], output_labels)
            loss.backward()
            this_grad = get_grad_vector(self.net.parameters, self.grad_dims).unsqueeze(0)
            self.opt.step()
            
            # Update steps on the replayed samples from buffer
            if self.sampled_memory_node_ids is not None and len(self.sampled_memory_node_ids) > 0:
                random_batch_inds = shuffled_inds[b_index * effective_batch_size: b_index * effective_batch_size + effective_batch_size]
                random_batch_inds = random_batch_inds.tolist()
                
                # random_batch_inds are indices in the buffer, which correspond to node indices in aux_g
                # (since aux_g is created from buffer nodes in order)
                batch_node_indices = torch.tensor(random_batch_inds, dtype=torch.long)
                if args.cuda:
                    batch_node_indices = batch_node_indices.to(device='cuda:{}'.format(args.gpu))
                
                # Sample blocks for buffer nodes
                _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, batch_node_indices)
                batch_x = aux_blocks[0].srcdata['feat']
                # Get labels from aux_labels using indices
                if isinstance(self.aux_labels, torch.Tensor):
                    batch_y = self.aux_labels[random_batch_inds]
                else:
                    batch_y = torch.tensor([self.aux_labels[i] for i in random_batch_inds], dtype=torch.long)
                if args.cuda:
                    batch_y = batch_y.to(device='cuda:{}'.format(args.gpu))
                
                self.net.zero_grad()
                output_aux, _ = self.net.forward_batch(aux_blocks, batch_x)
                if isinstance(output_aux, tuple):
                    output_aux = output_aux[0]
                loss_aux = self.ce(output_aux[:, self.offset1:self.offset2], batch_y)
                loss_aux.backward()
                self.opt.step()
                
                b_index += 1
                if b_index * effective_batch_size >= len(self.sampled_memory_node_ids):
                    b_index = 0
        
        # Memory update: when ring buffer is full
        if self.mem_cnt == self.n_memories:
            self.net.eval()
            
            if self.sampled_memory_node_ids is not None and self.n_sampled_memories <= len(self.sampled_memory_node_ids):
                # Buffer is full - decide on replacement
                effective_batch_size = min(self.n_constraints, len(self.sampled_memory_node_ids))
                batch_sim = self.get_batch_sim(args, g, features, labels, effective_batch_size)
                
                if batch_sim < self.sim_th:
                    
                    mem_node_ids = self.memory_node_ids.copy()
                    mem_labs = self.memory_labs.copy()
                    
                    # norm sim
                    cos_scores = self.sampled_memory_cos.clone()
                    cos_scores[torch.isnan(cos_scores)] = 0.0
                    
                    cos_min = torch.min(cos_scores)
                    cos_max = torch.max(cos_scores)
                    cos_range = cos_max - cos_min
                    
                    if cos_range < 1e-6:
                        # All values are the same, use uniform distribution
                        buffer_sim = torch.ones_like(cos_scores) / len(cos_scores)
                    else:
                        # Normalize to [0, 1]
                        buffer_sim = (cos_scores - cos_min) / (cos_range + 1e-6)
                        # buffer_sim = torch.clamp(buffer_sim, min=0.0, max=1.0)
                        # # Replace any remaining inf/nan with 0
                        # buffer_sim[torch.isinf(buffer_sim) | torch.isnan(buffer_sim)] = 0.0
                        # # Normalize to sum to 1 for multinomial
                        # buffer_sim = buffer_sim / (buffer_sim.sum() + 1e-8)
                    
                    # Draw candidates for replacement from the buffer
                    index = torch.multinomial(buffer_sim, len(mem_node_ids), replacement=False)
                    index = index.tolist()
                    
                    # Estimate similarity of each sample in received batch
                    batch_item_sim = self.get_each_batch_sample_sim(args, g, features, labels)
                    
                    # # Scale similarity scores to [0, 1] for probability
                    # batch_sim_scaled = batch_item_sim.clone()
                    # batch_sim_scaled[torch.isinf(batch_sim_scaled) | torch.isnan(batch_sim_scaled)] = 0.0
                    # scaled_batch_item_sim = torch.clamp((batch_sim_scaled + 1) / 2, min=0.0, max=1.0).unsqueeze(1).clone()
                    
                    # buffer_cos_selected = self.sampled_memory_cos[index].clone()
                    # buffer_cos_selected[torch.isinf(buffer_cos_selected) | torch.isnan(buffer_cos_selected)] = 0.0
                    # buffer_repl_batch_sim = torch.clamp((buffer_cos_selected + 1) / 2, min=0.0, max=1.0).unsqueeze(1).clone()
                    scaled_batch_item_sim=((batch_item_sim + 1) / 2).unsqueeze(1).clone()
                    buffer_repl_batch_sim=((self.sampled_memory_cos[index] + 1) / 2).unsqueeze(1).clone()
                    
                    # # Combine probabilities and normalize
                    # combined_probs = torch.cat((scaled_batch_item_sim, buffer_repl_batch_sim), dim=1)
                    # # Normalize each row to sum to 1
                    # combined_probs = combined_probs / (combined_probs.sum(dim=1, keepdim=True) + 1e-8)
                    
                    # # Draw an event to decide on replacement decision
                    # outcome = torch.multinomial(combined_probs, 1, replacement=False)
                    outcome=torch.multinomial(torch.cat((scaled_batch_item_sim,buffer_repl_batch_sim), dim=1), 1, replacement=False)#
                    
                    # Replace samples with outcome = 1
                    added_indx = torch.arange(end=batch_item_sim.size(0))
                    sub_index = outcome.squeeze(1).byte()
                    
                    for idx, replace in enumerate(sub_index):
                        if replace:
                            buffer_idx = index[idx]
                            self.sampled_memory_node_ids[buffer_idx] = mem_node_ids[idx]
                            self.sampled_memory_labs[buffer_idx] = mem_labs[idx]
                            self.sampled_memory_cos[buffer_idx] = batch_item_sim[idx].clone()
                    
                    # Update aux graph
                    self._update_aux_graph(g, features, labels)
            else:
                # Add new samples to the buffer
                added_inds = list(range(len(self.memory_node_ids)))
                
                # First buffer insertion
                if self.sampled_memory_node_ids is None:
                    self.sampled_memory_node_ids = self.memory_node_ids.copy()
                    self.sampled_memory_labs = self.memory_labs.copy()
                    self.sampled_memory_cos = torch.zeros(len(added_inds)) + 0.1
                else:
                    effective_batch_size = min(self.n_constraints, len(self.sampled_memory_node_ids))
                    self.get_batch_sim(args, g, features, labels, effective_batch_size)  # Draw random samples from buffer
                    this_sampled_memory_cos = self.get_each_batch_sample_sim(args, g, features, labels).clone()
                    
                    self.sampled_memory_cos = torch.cat((self.sampled_memory_cos, this_sampled_memory_cos.clone()), dim=0)
                    self.sampled_memory_node_ids.extend(self.memory_node_ids)
                    self.sampled_memory_labs.extend(self.memory_labs)
                
                # Update aux graph
                self._update_aux_graph(g, features, labels)
            
            # self.print_buffer_stats()
            self.mem_cnt = 0
            self.memory_node_ids = []
            self.memory_labs = []
            self.net.train()

    def _update_aux_graph(self, g, features, labels):
        """Update the auxiliary graph for buffer samples using the full graph"""
        if self.sampled_memory_node_ids is None or len(self.sampled_memory_node_ids) == 0:
            return
        
        if self.full_graph is None:
            print("Warning: Full graph not available, cannot update aux graph")
            return
        
        # Create subgraph with buffer nodes from the full graph
        # sampled_memory_node_ids stores original node IDs in the full graph
        buffer_node_ids_tensor = torch.tensor(self.sampled_memory_node_ids, dtype=torch.long)
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
            self.aux_labels = torch.tensor(self.sampled_memory_labs, dtype=torch.long)
            if self.cuda:
                self.aux_labels = self.aux_labels.to(device='cuda:{}'.format(self.gpu))
        except Exception as e:
            # If subgraph creation fails, skip update
            print(f"Warning: Could not update aux graph: {e}")
            pass

    def get_batch_sim(self, args, g, features, labels, effective_batch_size):
        """Estimate similarity score for the received samples to randomly drawn samples from buffer"""
        b_index = 0
        self.mem_grads = None
        shuffled_inds = torch.randperm(len(self.sampled_memory_node_ids))
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        
        for iter_i in range(int(self.rn)):
            random_batch_inds = shuffled_inds[b_index * effective_batch_size: b_index * effective_batch_size + effective_batch_size]
            random_batch_inds = random_batch_inds.tolist()
            
            # random_batch_inds are indices in the buffer, which correspond to node indices in aux_g
            # (since aux_g is created from buffer nodes in order)
            batch_node_indices = torch.tensor(random_batch_inds, dtype=torch.long)
            if args.cuda:
                batch_node_indices = batch_node_indices.to(device='cuda:{}'.format(args.gpu))
            
            _, _, aux_blocks = nb_sampler.sample_blocks(self.aux_g, batch_node_indices)
            batch_x = aux_blocks[0].srcdata['feat']
            # Get labels from aux_labels using indices
            if isinstance(self.aux_labels, torch.Tensor):
                batch_y = self.aux_labels[random_batch_inds]
            else:
                batch_y = torch.tensor([self.aux_labels[i] for i in random_batch_inds], dtype=torch.long)
            if args.cuda:
                batch_y = batch_y.to(device='cuda:{}'.format(args.gpu))
            
            self.net.zero_grad()
            output_aux, _ = self.net.forward_batch(aux_blocks, batch_x)
            if isinstance(output_aux, tuple):
                output_aux = output_aux[0]
            loss_aux = self.ce(output_aux[:, self.offset1:self.offset2], batch_y)
            loss_aux.backward()
            self.mem_grads = add_memory_grad(self.net.parameters, self.mem_grads, self.grad_dims)
            
            b_index += 1
            if b_index * effective_batch_size >= len(self.sampled_memory_node_ids):
                break
        
        # Compute gradient for current batch
        # memory_node_ids stores original node IDs, need to use full graph
        if self.full_graph is None:
            # Fallback: use current subgraph (should not happen in normal case)
            graph_to_use = g
            memory_node_ids_tensor = torch.tensor(self.memory_node_ids)
        else:
            # Use full graph with original node IDs
            graph_to_use = self.full_graph.to(device='cuda:{}'.format(args.gpu)) if args.cuda else self.full_graph
            memory_node_ids_tensor = torch.tensor(self.memory_node_ids, dtype=torch.long)
            if args.cuda:
                memory_node_ids_tensor = memory_node_ids_tensor.to(device='cuda:{}'.format(args.gpu))
        
        self.net.zero_grad()
        _, _, blocks = nb_sampler.sample_blocks(graph_to_use, memory_node_ids_tensor)
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)
        if isinstance(output, tuple):
            output = output[0]
        output_labels = torch.tensor(self.memory_labs, dtype=torch.long)
        if args.cuda:
            output_labels = output_labels.to(device='cuda:{}'.format(args.gpu))
        
        loss = self.ce(output[:, self.offset1:self.offset2], output_labels)
        loss.backward()
        this_grad = get_grad_vector(self.net.parameters, self.grad_dims).unsqueeze(0)
        
        # Compute cosine similarity and handle edge cases
        sim_matrix = self.cosine_similarity(self.mem_grads, this_grad)
        # Get max similarity, handling empty or invalid cases
        if sim_matrix.numel() == 0:
            batch_sim = torch.tensor(0.0)
        else:
            batch_sim = torch.max(sim_matrix)
            # Ensure the result is finite
            if torch.isinf(batch_sim) or torch.isnan(batch_sim):
                batch_sim = torch.tensor(0.0)
        
        return batch_sim.item() if isinstance(batch_sim, torch.Tensor) else batch_sim

    def get_each_batch_sample_sim(self, args, g, features, labels):
        """Estimate the similarity of each sample in the received batch to the randomly drawn samples from the buffer"""
        cosine_sim = torch.zeros(len(self.memory_node_ids))
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        
        # Use full graph if available, otherwise use current subgraph
        if self.full_graph is None:
            graph_to_use = g
        else:
            graph_to_use = self.full_graph.to(device='cuda:{}'.format(args.gpu)) if args.cuda else self.full_graph
        
        for item_index, node_id in enumerate(self.memory_node_ids):
            # node_id is original node ID in full graph
            node_id_tensor = torch.tensor([node_id], dtype=torch.long)
            if args.cuda:
                node_id_tensor = node_id_tensor.to(device='cuda:{}'.format(args.gpu))
            
            self.net.zero_grad()
            _, _, blocks = nb_sampler.sample_blocks(graph_to_use, node_id_tensor)
            input_features = blocks[0].srcdata['feat']
            output, _ = self.net.forward_batch(blocks, input_features)
            if isinstance(output, tuple):
                output = output[0]
            
            label_idx = self.memory_labs[item_index]
            label_tensor = torch.tensor([label_idx], dtype=torch.long)
            if args.cuda:
                label_tensor = label_tensor.to(device='cuda:{}'.format(args.gpu))
            
            ptloss = self.ce(output[:, self.offset1:self.offset2], label_tensor)
            ptloss.backward()
            
            # Add the new grad to the memory grads and add its cosine similarity
            this_grad = get_grad_vector(self.net.parameters, self.grad_dims).unsqueeze(0)
            sim_matrix = self.cosine_similarity(self.mem_grads, this_grad)
            # Get max similarity, handling edge cases
            if sim_matrix.numel() == 0:
                sim_value = 0.0
            else:
                sim_value = torch.max(sim_matrix)
                # Ensure the result is finite
                if torch.isinf(sim_value) or torch.isnan(sim_value):
                    sim_value = 0.0
                sim_value = sim_value.item() if isinstance(sim_value, torch.Tensor) else sim_value
            cosine_sim[item_index] = sim_value
        
        return cosine_sim

