import torch
import dgl
from .sem_utils import ReservoirSEM, ByClassReservoirSEM


class NET(torch.nn.Module):
    """
    SEM (Subgraph Episodic Memory) baseline adapted for the TFOCGL task-free
    online setting.

    Based on:
        "Ricci Curvature-Based Graph Sparsification for Continual Graph
         Representation Learning", Zhang et al., IEEE TNNLS 2023.

    Key difference from SSM (ICDM 2022):
        The subgraph context edges are chosen by *Ricci curvature proxy*
        (neighbourhood overlap) rather than randomly, keeping the most
        geometrically informative neighbours for each center node.

    The online adaptation mirrors ssm_model.py exactly:
    - Reservoir sampling maintains a uniform distribution over all seen nodes.
    - The buffer is updated once per epoch (at the epoch boundary).
    - Replay loss is added to the current-batch loss before each gradient step.

    Parameters
    ----------
    model : torch.nn.Module
        Backbone GNN (GCN / GAT / GIN).
    args : Namespace
        Experiment configuration.  Reads ``args.ssmer_args`` for:
        - 'budget'            : total node-slot budget for the buffer
        - 'memory_proportion' : replay batch size multiplier
        - 'nei_budget'        : fanout tuple for Ricci sparsification
    """

    def __init__(self, model, args):
        super(NET, self).__init__()

        self.net = model
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.ce = torch.nn.functional.cross_entropy

        budget = int(args.ssmer_args['budget'][0])
        self.memory_proportion = int(args.ssmer_args['memory_proportion'][0])
        nei_budget = args.ssmer_args['nei_budget'][0]

        self.buffer = ReservoirSEM(budget, nei_budget)

        self.seen_classes = []
        self.offset1 = 0
        self.offset2 = 0
        self.epochs = 0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_nb_sampler(self, args):
        if args.sample_nbs:
            return dgl.dataloading.NeighborSampler(args.n_nbs_sample)
        return dgl.dataloading.MultiLayerFullNeighborSampler(
            len(self.net.gat_layers))

    def _current_loss(self, output, labels, offset1=None, offset2=None):
        if offset1 is not None and offset2 is not None:
            return self.ce(output[:, offset1:offset2], labels)
        return self.ce(output, labels)

    def _replay_loss(self, nb_sampler, args, offset1=None, offset2=None):
        if len(self.buffer) == 0:
            return None

        n_samples = min(
            len(self.buffer),
            len(self.buffer.aux_labels) * self.memory_proportion,
        )
        n_samples = max(1, int(n_samples))

        aux_subgraphs, aux_labels = self.buffer.sample(n_samples)
        if not aux_subgraphs:
            return None

        batched = dgl.batch(aux_subgraphs)
        target_ids = torch.nonzero(
            batched.ndata['target'], as_tuple=False).squeeze()
        if target_ids.dim() == 0:
            target_ids = target_ids.unsqueeze(0)

        _, _, aux_blocks = nb_sampler.sample_blocks(batched, target_ids)
        aux_feat = aux_blocks[0].srcdata['feat']
        aux_out, _ = self.net.forward_batch(aux_blocks, aux_feat)
        if isinstance(aux_out, tuple):
            aux_out = aux_out[0]

        return self._current_loss(aux_out, aux_labels, offset1, offset2)

    # ------------------------------------------------------------------
    # Task-free online setting  (tfo / tf)
    # ------------------------------------------------------------------

    def observe(self, args, g, features, labels, train_ids):
        """
        Online update step — task-free setting.

        Parameters
        ----------
        args : Namespace
        g : DGLGraph
            Subgraph for the current streaming batch.
        features : Tensor
            Node feature matrix.
        labels : Tensor
            Node labels.
        train_ids : Tensor
            Indices of trainable nodes in the current batch.
        """
        self.net.train()
        self.epochs += 1
        last_epoch = self.epochs % args.epochs

        self.net.zero_grad()
        nb_sampler = self._make_nb_sampler(args)

        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_feat = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_feat)
        if isinstance(output, tuple):
            output = output[0]

        output_labels = labels[train_ids]
        loss = self.ce(output, output_labels)

        replay = self._replay_loss(nb_sampler, args)
        if replay is not None:
            loss = loss + replay

        loss.backward()
        self.opt.step()

        if last_epoch == 0:
            self.buffer.update(blocks, output_labels)

    # ------------------------------------------------------------------
    # Task-free online class-incremental setting  (tfocis)
    # ------------------------------------------------------------------

    def observe_cis(self, args, g, features, labels, train_ids):
        """
        Online update step — class-incremental (task-free) setting.

        Maintains an expanding output mask over classes seen so far,
        consistent with the class-IL protocol used by ssm_model.py.
        """
        self.net.train()
        self.epochs += 1
        last_epoch = self.epochs % args.epochs

        # Expand seen-class set and align output mask to even boundary
        for lbl in labels[train_ids].unique():
            if lbl not in self.seen_classes:
                self.seen_classes.append(lbl)
        self.offset2 = max(self.seen_classes).item() + 1
        if self.offset2 % 2 != 0:
            self.offset2 += 1

        self.net.zero_grad()
        nb_sampler = self._make_nb_sampler(args)

        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_feat = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_feat)
        if isinstance(output, tuple):
            output = output[0]

        output_labels = labels[train_ids]
        loss = self.ce(
            output[:, self.offset1:self.offset2], output_labels)

        replay = self._replay_loss(
            nb_sampler, args, self.offset1, self.offset2)
        if replay is not None:
            loss = loss + replay

        loss.backward()
        self.opt.step()

        if last_epoch == 0:
            self.buffer.update(blocks, output_labels)

    def forward(self, features):
        return self.net(features)
