import torch
import dgl
from collections import defaultdict


def build_adjacency(block):
    """
    Build adjacency dicts from a DGL block's edge list.

    In DGL bipartite blocks, dst node local-id i == src node local-id i
    (dst nodes occupy the first num_dst slots in the src space).
    This means N_in(d) (src-space indices) and N_out(s) (dst-space indices)
    can be directly intersected as integer sets to find common neighbors.

    Returns
    -------
    dst_to_srcs : dict[int -> set[int]]
        Maps each dst local-id to the set of src local-ids that send to it.
    src_to_dsts : dict[int -> set[int]]
        Maps each src local-id to the set of dst local-ids it sends to.
    """
    src_ids, dst_ids = block.edges()
    src_list = src_ids.cpu().tolist()
    dst_list = dst_ids.cpu().tolist()

    dst_to_srcs = defaultdict(set)
    src_to_dsts = defaultdict(set)
    for s, d in zip(src_list, dst_list):
        dst_to_srcs[d].add(s)
        src_to_dsts[s].add(d)
    return dst_to_srcs, src_to_dsts


def _ricci_score(s, d, dst_to_srcs, src_to_dsts):
    """
    Compute Ollivier-Ricci curvature surrogate for edge (s -> d).

    Proxy based on neighborhood overlap (Lin-Lu-Yau type):
        kappa(s, d) = (|N_in(d) ∩ N_out(s)| + 1) / (max(|N_in(d)|, |N_out(s)|) + 1)

    Higher score => edge lies in a more "clique-like" (convex) region =>
    the neighbor s is more redundantly connected to d's neighborhood =>
    more informative for message passing and worth keeping.
    """
    N_in_d = dst_to_srcs.get(d, set())   # src-space neighbors of dst d
    N_out_s = src_to_dsts.get(s, set())  # dst-space neighbors of src s
    # dst local-id == src local-id in DGL blocks, so the two sets share the
    # same integer domain and their intersection yields valid common neighbors.
    common = len(N_in_d & N_out_s)
    denom = max(len(N_in_d), len(N_out_s)) + 1
    return (common + 1) / denom


def sparsify_blocks_ricci(blocks, target_node, fanouts):
    """
    Build a sparsified subgraph memory for *target_node* using Ricci
    curvature-based neighbor selection (SEM, TNNLS 2023).

    Mirrors the structure of SSM's ``sparsify_blocks`` but replaces random
    sampling with top-k selection by Ricci curvature proxy score.  Edges
    with higher curvature (more common neighbors between src and dst) are
    kept, as they carry more aggregated neighbourhood information.

    Parameters
    ----------
    blocks : list of DGL blocks
        Output of a neighbor sampler; blocks[0] is the input-closest layer.
    target_node : int
        Local dst-node index in blocks[-1] for the node we store.
    fanouts : list of int
        Maximum number of neighbors to retain per hop (same order as blocks).

    Returns
    -------
    new_graph : DGLGraph
        Sparsified subgraph with 'feat' and 'target' node features.
        'target' is True for the center node, False for context nodes.
    """
    # Align fanout list with block order (blocks iterate output→input)
    if len(fanouts) == len(blocks):
        fanouts_reversed = list(fanouts[::-1])
    else:
        fanouts_reversed = list(fanouts[::-1]) if fanouts else [10] * len(blocks)
        while len(fanouts_reversed) < len(blocks):
            fanouts_reversed.append(fanouts_reversed[-1] if fanouts_reversed else 10)

    device = blocks[-1].device
    current_target_nodes = torch.tensor([target_node], device=device)

    # Initialise stored subgraph with the center (target) node only
    new_graph = dgl.graph(([], []), num_nodes=0, device=device)
    new_graph.add_nodes(1, {
        'feat': blocks[-1].dstdata['feat'][current_target_nodes],
        'target': torch.ones(1, dtype=torch.bool, device=device),
    })
    node_mapping = {target_node: 0}  # block local-id -> new_graph node-id
    next_node_index = 1

    for i in range(len(blocks) - 1, -1, -1):
        block = blocks[i]
        fanout_idx = len(blocks) - 1 - i
        fanout = (fanouts_reversed[fanout_idx]
                  if fanout_idx < len(fanouts_reversed)
                  else fanouts_reversed[-1])

        src_nodes, dst_nodes = block.edges()

        # Keep only edges whose dst is one of the current target nodes
        mask = torch.isin(dst_nodes, current_target_nodes)
        connected_src = src_nodes[mask]
        connected_dst = dst_nodes[mask]

        if len(connected_src) == 0:
            current_target_nodes = connected_src
            continue

        if len(connected_src) > fanout:
            # ------------------------------------------------------------------
            # Ricci curvature-based top-k selection
            # Build block adjacency once; O(|E_block|) pre-processing.
            # ------------------------------------------------------------------
            dst_to_srcs, src_to_dsts = build_adjacency(block)

            scores = torch.tensor(
                [_ricci_score(s, d,
                              dst_to_srcs, src_to_dsts)
                 for s, d in zip(connected_src.cpu().tolist(),
                                 connected_dst.cpu().tolist())],
                dtype=torch.float,
            )
            top_k = torch.topk(scores, fanout).indices
            sampled_src = connected_src[top_k]
            sampled_dst = connected_dst[top_k]
        else:
            sampled_src = connected_src
            sampled_dst = connected_dst

        # Add newly discovered nodes and their features to the stored subgraph
        for src_node in sampled_src:
            nid = src_node.item()
            if nid not in node_mapping:
                node_mapping[nid] = next_node_index
                new_graph.add_nodes(1)
                new_graph.ndata['feat'][next_node_index] = block.srcdata['feat'][src_node]
                new_graph.ndata['target'][next_node_index] = torch.zeros(
                    1, dtype=torch.bool, device=device)
                next_node_index += 1

        remapped_src = torch.tensor(
            [node_mapping[s.item()] for s in sampled_src], device=device)
        remapped_dst = torch.tensor(
            [node_mapping[d.item()] for d in sampled_dst], device=device)
        new_graph.add_edges(remapped_src, remapped_dst)

        # The sampled src nodes become the targets for the next (outer) hop
        current_target_nodes = sampled_src.unique()

    # Self-loops ensure every node aggregates its own feature
    new_graph.add_edges(new_graph.nodes(), new_graph.nodes())
    return new_graph


# ---------------------------------------------------------------------------
# Reservoir buffer — identical interface to ReservoirSSM in ssm_utils.py
# but calls sparsify_blocks_ricci instead of the random-based sparsify_blocks
# ---------------------------------------------------------------------------

class ReservoirSEM:
    """
    Reservoir-sampled Subgraph Episodic Memory (SEM).

    Stores sparsified computation subgraphs whose context edges are selected
    by Ricci curvature proxy rather than random or degree-based sampling.
    Reservoir sampling guarantees a uniform distribution over all nodes seen.

    Parameters
    ----------
    budget : int
        Total number of center nodes (subgraphs) to keep.
    nei_budget : tuple / list of int
        Fanout per hop, passed to sparsify_blocks_ricci.
    """

    def __init__(self, budget, nei_budget):
        # Effective capacity: each subgraph costs ~sum(nei_budget) node slots,
        # so we keep budget // sum(nei_budget) subgraphs (mirrors ReservoirSSM).
        self.budget = max(1, budget // max(sum(nei_budget), 1))
        self.nei_budget = nei_budget
        self.aux_subgraphs = []
        self.aux_labels = []
        self.n_seen_examples = 0

    def __len__(self):
        return len(self.aux_labels)

    def sample(self, n_samples):
        """Return a random subset of stored (subgraph, label) pairs."""
        n = min(n_samples, len(self))
        indices = torch.randperm(len(self))[:n]
        return (
            [self.aux_subgraphs[i] for i in indices],
            self.aux_labels[indices],
        )

    def update(self, blocks, labels):
        """
        Incorporate the current mini-batch into the reservoir.

        For each node in the batch:
        - If space remains, store unconditionally.
        - Otherwise replace a random existing entry with probability
          budget / (n_seen + i), preserving the uniform distribution.
        """
        place_left = max(0, self.budget - len(self))
        n_nodes = len(labels)

        if place_left:
            offset = min(place_left, n_nodes)
            new_subgraphs = [
                sparsify_blocks_ricci(blocks, i, self.nei_budget)
                for i in range(offset)
            ]
            if len(self.aux_labels) > 0:
                self.aux_subgraphs.extend(new_subgraphs)
                self.aux_labels = torch.cat(
                    [self.aux_labels, labels[:offset]], dim=0)
            else:
                self.aux_subgraphs = new_subgraphs
                self.aux_labels = labels[:offset].clone()

            # Reservoir replacement for nodes beyond the initial fill
            for i in range(offset, n_nodes):
                j = torch.randint(0, self.n_seen_examples + i, (1,)).item()
                if j < self.budget:
                    self.aux_subgraphs[j] = sparsify_blocks_ricci(
                        blocks, i, self.nei_budget)
                    self.aux_labels[j] = labels[i]
        else:
            for i in range(n_nodes):
                j = torch.randint(0, self.n_seen_examples + i, (1,)).item()
                if j < self.budget:
                    self.aux_subgraphs[j] = sparsify_blocks_ricci(
                        blocks, i, self.nei_budget)
                    self.aux_labels[j] = labels[i]

        self.n_seen_examples += n_nodes


class ByClassReservoirSEM:
    """
    Per-class variant of ReservoirSEM for balanced class coverage.
    Mirrors ByClassReservoirSSM from ssm_utils.py.
    """

    def __init__(self, budget, num_classes, nei_budget):
        self.budget = budget
        self.num_classes = num_classes
        self.aux_buffers = [
            ReservoirSEM(budget // num_classes, nei_budget)
            for _ in range(num_classes)
        ]

    def __len__(self):
        return sum(len(b) for b in self.aux_buffers)

    def sample(self, n_samples):
        n = min(n_samples, len(self))
        global_indices = torch.randperm(len(self))[:n]

        offsets = torch.cumsum(
            torch.tensor([0] + [len(b) for b in self.aux_buffers]), dim=0)

        sampled_subgraphs = []
        sampled_labels = []
        for idx, buf in enumerate(self.aux_buffers):
            in_buf = (global_indices >= offsets[idx]) & (
                global_indices < offsets[idx + 1])
            local_ids = global_indices[in_buf] - offsets[idx]
            if len(local_ids) > 0:
                sampled_subgraphs.extend(
                    [buf.aux_subgraphs[j] for j in local_ids.tolist()])
                sampled_labels.append(buf.aux_labels[local_ids])

        if sampled_labels:
            sampled_labels = torch.cat(sampled_labels, dim=0)
        return sampled_subgraphs, sampled_labels

    def update(self, blocks, labels):
        for cls_id in range(self.num_classes):
            mask = (labels == cls_id)
            if mask.sum() > 0:
                self.aux_buffers[cls_id].update(blocks, labels[mask])
