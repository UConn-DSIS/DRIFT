import torch
import dgl


class ReservoirSSM:
    def __init__(self, budget, nei_budget):
        self.budget = budget // sum(nei_budget)
        self.aux_subgraphs = []
        self.aux_labels = []
        self.n_seen_examples = 0
        self.nei_budget = nei_budget

    def __len__(self):
        return len(self.aux_labels)

    def sample(self, n_samples):
        sampled_indices = torch.randperm(len(self))[:n_samples]
        return [self.aux_subgraphs[i] for i in sampled_indices], self.aux_labels[sampled_indices]

    def update(self, blocks, labels):
        place_left = max(0, self.budget - len(self))
        n_nodes = len(labels)
        if place_left:
            offset = min(place_left, n_nodes)
            if offset > 0:
                if len(self.aux_labels) > 0:
                    self.aux_subgraphs.extend([sparsify_blocks(blocks, i, self.nei_budget) for i in range(offset)])
                    self.aux_labels = torch.cat([self.aux_labels, labels[:offset]], dim=0)
                else:
                    self.aux_subgraphs = [sparsify_blocks(blocks, i, self.nei_budget) for i in range(offset)]
                    self.aux_labels = labels[:offset]
            if offset < n_nodes:
                for i in range(offset, n_nodes):
                    j = torch.randint(0, self.n_seen_examples + i, (1,))
                    if j < self.budget:
                        self.aux_subgraphs[j] = sparsify_blocks(blocks, i, self.nei_budget)
                        self.aux_labels[j] = labels[i]
        else:
            for i in range(n_nodes):
                j = torch.randint(0, self.n_seen_examples + i, (1,))
                if j < self.budget:
                    self.aux_subgraphs[j] = sparsify_blocks(blocks, i, self.nei_budget)
                    self.aux_labels[j] = labels[i]
        self.n_seen_examples += n_nodes

class ByClassReservoirSSM:
    def __init__(self, budget, num_classes, nei_budget):
        self.budget = budget
        self.num_classes = num_classes
        self.aux_buffers = [ReservoirSSM(budget//num_classes, nei_budget) for _ in range(num_classes)]
    
    def __len__(self):
        return sum([len(buffer) for buffer in self.aux_buffers])
    
    def sample(self, n_samples):
        sampled_subgraphs = []
        sampled_labels = []

        global_indices = torch.randperm(len(self))[:n_samples]

        buffer_offsets = [0] + [len(buffer) for buffer in self.aux_buffers]
        buffer_offsets = torch.cumsum(torch.tensor(buffer_offsets), dim=0)

        for i, buffer in enumerate(self.aux_buffers):
            in_buffer = (global_indices >= buffer_offsets[i]) & (global_indices < buffer_offsets[i + 1])
            local_indices = global_indices[in_buffer] - buffer_offsets[i]

            if len(local_indices) > 0:
                sampled_subgraphs.extend([buffer.aux_subgraphs[j] for j in local_indices])
                sampled_labels.append(buffer.aux_labels[local_indices])

        if sampled_labels:
            sampled_labels = torch.cat(sampled_labels, dim=0)

        return sampled_subgraphs, sampled_labels

    def update(self, blocks, labels):
        for i in range(self.num_classes):
            mask = labels == i
            if mask.sum() > 0:
                self.aux_buffers[i].update(blocks, labels[mask])


def sparsify_blocks(blocks, target_node, fanouts):
    """
    Sparsify blocks to create a subgraph memory for a target node.
    
    :param blocks: List of DGL blocks from sampler (blocks[-1] is the output layer)
    :param target_node: Index of target node in blocks[-1].dstdata
    :param fanouts: List of fanouts for each hop (should match the order of blocks, from output to input)
    """
    # Reverse fanouts to match the order of blocks (from output to input)
    # fanouts is typically given from input to output, but blocks are from output to input
    if len(fanouts) == len(blocks):
        # If fanouts length matches blocks, assume they're in the same order
        fanouts_reversed = fanouts[::-1]
    else:
        # If not matching, reverse and pad/truncate as needed
        fanouts_reversed = fanouts[::-1] if len(fanouts) > 0 else [10] * len(blocks)
        while len(fanouts_reversed) < len(blocks):
            fanouts_reversed.append(fanouts_reversed[-1] if len(fanouts_reversed) > 0 else 10)

    current_target_nodes = torch.tensor([target_node], device=blocks[-1].device)

    # Initialize the new graph with only the target node
    new_graph = dgl.graph(([], []), num_nodes=0, device=blocks[-1].device)

    # Add the target node and its features
    new_graph.add_nodes(1, {'feat': blocks[-1].dstdata['feat'][current_target_nodes],
                            'target': torch.ones(1, dtype=torch.bool, device=blocks[-1].device)})

    # Dictionary to keep track of the new node indices
    node_mapping = {target_node: 0}
    next_node_index = 1

    for i in range(len(blocks) - 1, -1, -1):
        block = blocks[i]
        # Map block index to fanout index (blocks are from output to input)
        fanout_idx = len(blocks) - 1 - i
        fanout = fanouts_reversed[fanout_idx] if fanout_idx < len(fanouts_reversed) else fanouts_reversed[-1]

        # Get the source and destination nodes of the current block
        src_nodes, dst_nodes = block.edges()
        
        # Filter the source nodes that are connected to the current target nodes
        mask = torch.isin(dst_nodes, current_target_nodes)
        connected_src_nodes = src_nodes[mask]
        connected_dst_nodes = dst_nodes[mask]

        # Sample the source nodes randomly based on the fanout
        if len(connected_src_nodes) > fanout:
            perm = torch.randperm(len(connected_src_nodes))
            sampled_src_nodes = connected_src_nodes[perm[:fanout]]
            sampled_dst_nodes = connected_dst_nodes[perm[:fanout]]
        else:
            sampled_src_nodes = connected_src_nodes
            sampled_dst_nodes = connected_dst_nodes

        # Add the sampled nodes and their features to the new graph
        for src_node in sampled_src_nodes:
            if src_node.item() not in node_mapping:
                node_mapping[src_node.item()] = next_node_index
                new_graph.add_nodes(1)
                new_graph.ndata['feat'][next_node_index] = block.srcdata['feat'][src_node]
                new_graph.ndata['target'][next_node_index] = torch.zeros(1, dtype=torch.bool, device=blocks[-1].device)
                next_node_index += 1

        # Remap the indices for the edges
        remapped_src_nodes = torch.tensor([node_mapping[src_node.item()] for src_node in sampled_src_nodes], device=blocks[-1].device)
        remapped_dst_nodes = torch.tensor([node_mapping[dst_node.item()] for dst_node in sampled_dst_nodes], device=blocks[-1].device)

        # Connect the sampled nodes to the target nodes
        new_graph.add_edges(remapped_src_nodes, remapped_dst_nodes)

        # Update the target nodes for the next iteration
        current_target_nodes = sampled_src_nodes

    # add self loops
    new_graph.add_edges(new_graph.nodes(), new_graph.nodes())

    return new_graph    