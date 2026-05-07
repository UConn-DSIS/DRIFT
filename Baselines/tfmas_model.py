import torch
import dgl
import numpy as np

class NET(torch.nn.Module):
    """
        MAS baseline for NCGL tasks

        :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
        :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

        """
    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()
        self.reg = args.mas_args['memory_strength']

        # setup network
        self.net = model

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # loss detector
        self.loss_window = []
        self.loss_window_means = []
        self.loss_window_variances = []
        self.new_peak_detected = True
        # MAS regularization: list of 3 weights vectors as there are 3 layers.
        self.star_variables = []
        self.omegas = [] #initialize with 0 importance weights
        self.MAS_weight = 0.5
        self.count_updates = 0
        self.loss_window_length = 5
        self.loss_window_mean_threshold = 0.2,
        self.loss_window_variance_threshold = 0.1, 

        self.optpar = []
        self.fisher = []
        self.n_seen_examples = 0
        self.epochs = 0
        self.seen_classes = []

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

        if len(self.star_variables)!=0 and len(self.omegas)!=0:
            for pindex, p in enumerate(self.net.parameters()): 
                loss += self.MAS_weight/2.*torch.sum(self.omegas[pindex]*(p-self.star_variables[pindex])**2)

        loss.backward()
        self.opt.step()

        # add loss to loss_window and detect loss plateaus
        self.loss_window.append(loss.cpu().detach().numpy())
        if len(self.loss_window) > self.loss_window_length: 
            del self.loss_window[0]
        self.loss_window_mean = np.mean(self.loss_window)
        self.loss_window_variance = np.var(self.loss_window)
        #check the statistics of the current window
        if not self.new_peak_detected and self.loss_window_mean > self.last_loss_window_mean + np.sqrt(self.last_loss_window_variance):
            self.new_peak_detected = True 

        if self.loss_window_mean < self.loss_window_mean_threshold and self.loss_window_variance < self.loss_window_variance_threshold and self.new_peak_detected:
            self.count_updates+=1
            self.last_loss_window_mean = self.loss_window_mean
            self.last_loss_window_variance = self.loss_window_variance
            self.new_peak_detected = False
                        
            # calculate importance weights and update star_variables
            gradients=[0 for p in self.net.parameters()]
            self.net.zero_grad()
            output, _ = self.net.forward_batch(blocks, input_features)
            if isinstance(output, tuple):
                output = output[0]
            output.pow_(2)
            loss = output.mean()
            loss.backward()

            for pindex, p in enumerate(self.net.parameters()):
                g = p.grad.data.clone()
                gradients[pindex] += torch.abs(g)

            omegas_old = self.omegas[:]
            self.omegas = []
            self.star_variables = []
            for pindex, p in enumerate(self.net.parameters()):
                if len(omegas_old) != 0:
                    self.omegas.append(1/self.count_updates*gradients[pindex]+(1-1/self.count_updates)*omegas_old[pindex])
                else:
                    self.omegas.append(gradients[pindex])
                self.star_variables.append(p.data.clone().detach())
        
        self.loss_window_means.append(self.loss_window_mean)
        self.loss_window_variances.append(self.loss_window_variance)

    
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

        for label in labels[train_ids].unique():
            if label not in self.seen_classes:
                self.seen_classes.append(label)
        offset1, offset2 = 0, max(self.seen_classes)+1
        if offset2 % 2 != 0:
            offset2 += 1

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
        loss = self.ce(output[:, offset1:offset2], output_labels)

        if len(self.star_variables)!=0 and len(self.omegas)!=0:
            for pindex, p in enumerate(self.net.parameters()):
                # print(pindex, type(self.omegas[pindex]), self.omegas[pindex])
                loss += self.MAS_weight/2.*torch.sum(self.omegas[pindex]*(p-self.star_variables[pindex])**2)

        loss.backward()
        self.opt.step()

        # add loss to loss_window and detect loss plateaus
        self.loss_window.append(loss.cpu().detach().numpy())
        if len(self.loss_window) > self.loss_window_length: 
            del self.loss_window[0]
        self.loss_window_mean = np.mean(self.loss_window)
        self.loss_window_variance = np.var(self.loss_window)
        #check the statistics of the current window
        if not self.new_peak_detected and self.loss_window_mean > self.last_loss_window_mean + np.sqrt(self.last_loss_window_variance):
            self.new_peak_detected = True 

        if self.loss_window_mean < self.loss_window_mean_threshold and self.loss_window_variance < self.loss_window_variance_threshold and self.new_peak_detected:
            self.count_updates+=1
            self.last_loss_window_mean = self.loss_window_mean
            self.last_loss_window_variance = self.loss_window_variance
            self.new_peak_detected = False
                        
            # calculate importance weights and update star_variables
            gradients=[0 for p in self.net.parameters()]
            self.net.zero_grad()
            output, _ = self.net.forward_batch(blocks, input_features)
            if isinstance(output, tuple):
                output = output[0]
            output = output[:, offset1:offset2]
            output.pow_(2)
            loss = output.mean()
            loss.backward()

            for pindex, p in enumerate(self.net.parameters()):
                g = p.grad.data.clone()
                gradients[pindex] += torch.abs(g)

            omegas_old = self.omegas[:]
            self.omegas = []
            self.star_variables = []
            for pindex, p in enumerate(self.net.parameters()):
                if len(omegas_old) != 0:
                    self.omegas.append(1/self.count_updates*gradients[pindex]+(1-1/self.count_updates)*omegas_old[pindex])
                else:
                    self.omegas.append(gradients[pindex])
                self.star_variables.append(p.data.clone().detach())
        
        self.loss_window_means.append(self.loss_window_mean)
        self.loss_window_variances.append(self.loss_window_variance)