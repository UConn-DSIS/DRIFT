import torch
import dgl
import copy

class NET(torch.nn.Module):

    """
    Bare model baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 args):
        """
        The initialization of the baseline

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.
        """
        super(NET, self).__init__()

        # backbone model
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup loss
        self.ce = torch.nn.functional.cross_entropy

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
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param train_ids: The indices of the nodes participating in the training.

        """
        self.net.train()
        self.net.zero_grad()

        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        # few previous samples are accessible for message passing
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        output_labels = labels[train_ids]
        input_features = blocks[0].srcdata['feat']
        
        output, _ = self.net.forward_batch(blocks, input_features)

        if isinstance(output,tuple):
            output = output[0]
        loss = self.ce(output, output_labels)
        # print(f'labels range: {output_labels.min()} to {output_labels.max()}')
        loss.backward()
        self.opt.step()
    
    def observe_cis(self, args, g, features, labels, train_ids):
        """
        The method for learning the given tasks under the task free online cis setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param train_ids: The indices of the nodes participating in the training.
        """
        self.net.train()
        self.net.zero_grad()

        # Use only this batch's labels so CIS expansion matches sequential tfocis.
        # On tfo_gaussian, `g` is the merged subgraph: `labels.unique()` would mark
        # every dataset class as seen on the first step (wrong).
        for label in labels[train_ids].unique():
            li = int(label.item())
            if li not in self.seen_classes:
                self.seen_classes.append(li)
        self.offset2 = max(self.seen_classes) + 1
        if self.offset2 % 2 != 0:
            self.offset2 += 1

        nb_sampler = dgl.dataloading.NeighborSampler(args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(len(self.net.gat_layers))
        if args.cuda:
            train_ids = train_ids.to(device='cuda:{}'.format(args.gpu))

        # few previous samples are accessible for message passing
        _, _, blocks = nb_sampler.sample_blocks(g, train_ids)
        output_labels = labels[train_ids]
        input_features = blocks[0].srcdata['feat']
        output, _ = self.net.forward_batch(blocks, input_features)

        if isinstance(output,tuple):
            output = output[0]
        loss = self.ce(output[:, self.offset1:self.offset2], output_labels)
        loss.backward()
        self.opt.step()