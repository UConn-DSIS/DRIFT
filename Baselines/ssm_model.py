import torch
import dgl
from .ssm_utils import ReservoirSSM, ByClassReservoirSSM

class NET(torch.nn.Module):
    """
    SSM baseline for NCGL tasks adapted for TFOCGL setting
    Based on Sparsified Subgraph Memory for Continual Graph Representation Learning
    Adapted with reservoir sampling for online setting
    """

    def __init__(self, model, args):
        """
        The initialization of the SSM baseline

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param args: The arguments containing the configurations of the experiments
        """
        super(NET, self).__init__()

        # backbone model
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup loss
        self.ce = torch.nn.functional.cross_entropy

        # setup memory buffer
        budget = int(args.ssmer_args['budget'][0])
        self.memory_proportion = int(args.ssmer_args['memory_proportion'][0])
        nei_budget = args.ssmer_args['nei_budget'][0]
        
        # if args.dataset == 'Elliptic' or args.dataset == 'Elliptic-CL':
        #     self.buffer = ByClassReservoirSSM(budget, args.n_cls, nei_budget)
        # else:
        self.buffer = ReservoirSSM(budget, nei_budget)

        self.seen_classes = []
        self.offset1 = 0
        self.offset2 = 0
        self.epochs = 0

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

        self.net.zero_grad()
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        # Extract blocks for current batch
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)

        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        loss = self.ce(output, output_labels)

        # Compute auxiliary loss from buffer
        if len(self.buffer) > 0:
            n_samples = min(len(output_labels) * self.memory_proportion, len(self.buffer))
            aux_subgraphs, aux_labels = self.buffer.sample(n_samples)
            if len(aux_subgraphs) > 0:
                batched_graph = dgl.batch(aux_subgraphs)
                target_node_ids = torch.nonzero(batched_graph.ndata['target'], as_tuple=False).squeeze()
                if len(target_node_ids.shape) == 0:
                    target_node_ids = target_node_ids.unsqueeze(0)
                _, _, aux_blocks = nb_sampler.sample_blocks(batched_graph, target_node_ids)
                aux_features = aux_blocks[0].srcdata['feat']
                aux_output, _ = self.net.forward_batch(aux_blocks, aux_features)
                if isinstance(aux_output, tuple):
                    aux_output = aux_output[0]
                loss_aux = self.ce(aux_output, aux_labels)
                loss = loss + loss_aux

        loss.backward()
        self.opt.step()

        # Update buffer at the end of each epoch
        if last_epoch == 0:
            self.buffer.update(blocks, output_labels)

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
            if label not in self.seen_classes:
                self.seen_classes.append(label)
        self.offset2 = max(self.seen_classes) + 1
        if self.offset2 % 2 != 0:
            self.offset2 += 1

        self.net.zero_grad()
        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        # Extract blocks for current batch
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)

        if isinstance(output, tuple):
            output = output[0]
        output_labels = labels[train_ids]
        loss = self.ce(output[:, self.offset1:self.offset2], output_labels)

        # Compute auxiliary loss from buffer
        if len(self.buffer) > 0:
            n_samples = min(len(output_labels) * self.memory_proportion, len(self.buffer))
            aux_subgraphs, aux_labels = self.buffer.sample(n_samples)
            if len(aux_subgraphs) > 0:
                batched_graph = dgl.batch(aux_subgraphs)
                target_node_ids = torch.nonzero(batched_graph.ndata['target'], as_tuple=False).squeeze()
                if len(target_node_ids.shape) == 0:
                    target_node_ids = target_node_ids.unsqueeze(0)
                _, _, aux_blocks = nb_sampler.sample_blocks(batched_graph, target_node_ids)
                aux_features = aux_blocks[0].srcdata['feat']
                aux_output, _ = self.net.forward_batch(aux_blocks, aux_features)
                if isinstance(aux_output, tuple):
                    aux_output = aux_output[0]
                loss_aux = self.ce(aux_output[:, self.offset1:self.offset2], aux_labels)
                loss = loss + loss_aux

        loss.backward()
        self.opt.step()

        # Update buffer at the end of each epoch
        if last_epoch == 0:
            self.buffer.update(blocks, output_labels)
