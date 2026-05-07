import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
from dgl.data import CoraFullDataset, RedditDataset, AmazonCoBuyComputerDataset, RomanEmpireDataset
import copy
import collections
import pandas as pd
import os
from torch_geometric.io import fs


def get_nhop_neighborhood(graph, nodes, n_hops, device):
    # Ensure input nodes are a tensor
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes).to(device)

    all_nodes = nodes
    for _ in range(n_hops):
        # Get outgoing edges from current nodes
        u, v = graph.out_edges(all_nodes)
        neighbors = v.unique()

        # Combine current nodes and neighbors into a single list
        all_nodes = torch.cat([all_nodes, neighbors]).unique()

    return all_nodes.tolist()

class Continuum:

    def __init__(self, data, args):
        self.data = data[1][0]
        self.graphs = data[0]
        self.batch_size = args.batch_size
        n_tasks = len(self.data)
        args.n_tasks = n_tasks

        task_permutation = range(n_tasks)

        if args.shuffle_tasks == 'yes':
            task_permutation = torch.randperm(n_tasks).tolist()

        sample_permutations = []

        self.samples_per_task = args.samples_per_task
        n = 0
        for t in range(n_tasks):
            N=len(self.data[t])
            if self.samples_per_task > 0:
                n = min(self.samples_per_task, N)
            else:
                n = N
            print("*********Task",t,"Samples are",n)
            random.shuffle(self.data[t])
            p = self.data[t][0:n]
            sample_permutations.append(p)

        self.permutation = []

        for t in range(n_tasks):
            task_t = task_permutation[t]

            task_p = [[task_t, i] for i in sample_permutations[task_t]]
            random.shuffle(task_p)
            self.permutation += task_p

        self.length = len(self.permutation)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            t = self.permutation[self.current][0]
            j = []
            i = 0
            while (((self.current + i) < self.length) and
                   (self.permutation[self.current + i][0] == t) and
                   (i < self.batch_size)):
                j.append(self.permutation[self.current + i][1])
                i += 1
            self.current += i
            j = torch.LongTensor(j)
            sub_g = self.graphs[t]
            return sub_g, t, j

class TimeContinuum:
    """
    Continuum iterator for time incremental graph.
    """
    def __init__(self, data, args):
        """
        Initialize TimeContinuum
        
        Args:
            data: [subgraphs, [tasks_tr, tasks_te]] format from data_prepare_tem
            args: arguments containing batch_size, samples_per_task, etc.
        """
        self.graphs = data[0]  
        self.tasks_tr = data[1][0]
        
        self.batch_size = args.batch_size
        self.samples_per_task = args.samples_per_task
        self.n_time_tasks = len(self.graphs)
        
        self.task_sequence = list(range(self.n_time_tasks))
        
        # If shuffle_tasks is enabled, we can still shuffle task order
        # but samples within each task will remain time-ordered
        if hasattr(args, 'shuffle_tasks') and args.shuffle_tasks == 'yes':
            random.shuffle(self.task_sequence)
        
        self.task_samples = {}
        
        for task_id in range(self.n_time_tasks):
            train_ids = self.tasks_tr[task_id]
            # Only use training nodes for iteration
            all_task_nodes = train_ids
            
            if self.samples_per_task > 0:
                n_samples = min(self.samples_per_task, len(all_task_nodes))
                task_nodes = all_task_nodes[:n_samples]
            else:
                task_nodes = all_task_nodes
            
            self.task_samples[task_id] = task_nodes
            
            print(f"*********Time Task {task_id} Samples are {len(task_nodes)}")

        self.permutation = []
        for task_id in self.task_sequence:
            task_samples = self.task_samples[task_id]

            for sample_id in task_samples:
                self.permutation.append([task_id, sample_id])
        
        self.length = len(self.permutation)
        self.current = 0
        
        # print(f"TimeContinuum initialized: {self.length} total samples across {self.n_time_tasks} tasks")

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.current >= self.length:
            raise StopIteration
        else:
            t = self.permutation[self.current][0]
            j = []
            i = 0
            
            while ((self.current + i) < self.length and
                   self.permutation[self.current + i][0] == t and
                   i < self.batch_size):
                j.append(self.permutation[self.current + i][1])
                i += 1
            
            self.current += i
            
            j = torch.LongTensor(j)
            
            subgraph = self.graphs[t]
            
            return subgraph, t, j

    def get_task_progress(self, task_id):
        """Get progress information for a specific task"""
        if task_id not in self.task_samples:
            return None
        
        total_samples = len(self.task_samples[task_id])
        processed_samples = min(self.current, total_samples)
        
        return {
            'task_id': task_id,
            'total_samples': total_samples,
            'processed_samples': processed_samples,
            'progress': processed_samples / total_samples if total_samples > 0 else 0
        }


class incremental_graph_trans_(nn.Module):
    def __init__(self, dataset, n_cls):
        super().__init__()
        # transductive setting
        self.graph, self.labels = dataset[0]
        #self.graph = dgl.add_reverse_edges(self.graph)
        #self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['label'] = self.labels
        self.d_data = self.graph.ndata['feat'].shape[1]
        self.n_cls = n_cls
        self.n_nodes = self.labels.shape[0]
        self.tr_va_te_split = dataset[1]

    def get_graph(self, tasks_to_retain=[], node_ids=None):
        # get the partial graph
        # tasks-to-retain: classes retained in the partial graph
        node_ids_ = copy.deepcopy(node_ids)
        node_ids_retained = []
        ids_train_old, ids_test_old = [], []
        if len(tasks_to_retain) > 0:
            # retain nodes according to classes
            for t in tasks_to_retain:
                ids_train_old.extend(self.tr_va_te_split[t][0])
                # ids_valid_old.extend(self.tr_va_te_split[t][1])
                ids_test_old.extend(self.tr_va_te_split[t][2])
                node_ids_retained.extend(self.tr_va_te_split[t][0] + self.tr_va_te_split[t][2])
            subgraph_0 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True) # ! used eval nodes before, now revised
            if node_ids_ is None:
                subgraph = subgraph_0
        if node_ids_ is not None:
            # node_ids: [tr, te]
            [ids_train, ids_test] = node_ids_
            ids_train_old = copy.deepcopy(ids_train)
            ids_test_old = copy.deepcopy(ids_test)
            node_ids_retained.extend(ids_train_old + ids_test_old)
            if isinstance(node_ids_,list):
                subgraph_1 = dgl.node_subgraph(self.graph, node_ids_retained, store_ids=True)

            if len(tasks_to_retain)==0:
                subgraph = subgraph_1

        if len(tasks_to_retain) > 0 and node_ids is not None:
            raise ValueError("tasks_to_retain and node_ids cannot be both non-empty/None at the same time")

        old_ids = subgraph.ndata['_ID'].cpu()
        ids_train = [(old_ids == i).nonzero()[0][0].item() for i in ids_train_old]
        # ids_val = [(old_ids == i).nonzero()[0][0].item() for i in ids_valid_old]
        ids_test = [(old_ids == i).nonzero()[0][0].item() for i in ids_test_old]
        # node_ids_per_task_reordered = []
        # for c in tasks_to_retain:
        #     ids = (subgraph.ndata['label'] == c).nonzero()[:, 0].view(-1).tolist()
        #     node_ids_per_task_reordered.append(ids)
        subgraph = dgl.add_self_loop(subgraph)

        return subgraph, [ids_train, ids_test]

def train_valid_test_split(ids,ratio_valid_test):
    va_te_ratio = sum(ratio_valid_test)
    train_ids, va_te_ids = train_test_split(ids, test_size=va_te_ratio)
    return [train_ids] + train_test_split(va_te_ids, test_size=ratio_valid_test[1]/va_te_ratio)

class NodeLevelDataset(incremental_graph_trans_):
    def __init__(self,name='ogbn-arxiv',IL='class',default_split=False,ratio_valid_test=None,args=None):
        r""""
        name: name of the dataset
        IL: use task- or class-incremental setting
        default_split: if True, each class is split according to the splitting of the original dataset, which may cause the train-val-test ratio of different classes greatly different
        ratio_valid_test: in form of [r_val,r_test] ratio of validation and test set, train set ratio is directly calculated by 1-r_val-r_test
        """

        # return an incremental graph instance that can return required subgraph upon request
        if name[0:4] == 'ogbn':
            data = DglNodePropPredDataset(name, root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name in ['CoraFullDataset', 'CoraFull','corafull', 'CoraFull-CL','Corafull-CL']:
            data = CoraFullDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1)
        elif name in ['RomanEmpire', 'RomanEmpire-CL', 'romanempire']:
            data = RomanEmpireDataset()
            graph, label = data[0], data[0].dstdata['label'].view(-1, 1) 
        elif name in ['reddit','Reddit','Reddit-CL']:
            data = RedditDataset(self_loop=False)
            # graph, label = data.graph, data.labels.view(-1, 1)
            graph = data[0]
            label = graph.ndata['label']
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        elif name == 'Products-CL':
            data = DglNodePropPredDataset('ogbn-products', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
        else:
            print('invalid data name')
        n_cls = data.num_classes
        cls = [i for i in range(n_cls)]
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        cls_sizes = {c: len(cls_id_map[c]) for c in cls_id_map}
        for c in cls_sizes:
            if cls_sizes[c] < 2:
                cls.remove(c) # remove classes with less than 2 examples, which cannot be split into train, val, test sets
        cls_id_map = {i: list((label.squeeze() == i).nonzero().squeeze().view(-1, ).numpy()) for i in cls}
        n_cls = len(cls)
        if default_split:
            split_idx = data.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
                "test"].tolist()
            tr_va_te_split = {c: [list(set(cls_id_map[c]).intersection(set(train_idx))),
                                  list(set(cls_id_map[c]).intersection(set(valid_idx))),
                                  list(set(cls_id_map[c]).intersection(set(test_idx)))] for c in cls}

        elif not default_split:
            split_name = f'{args.data_path}/tr{round(1-ratio_valid_test[0]-ratio_valid_test[1],2)}_va{ratio_valid_test[0]}_te{ratio_valid_test[1]}_split_{name}.pkl'
            try:
                tr_va_te_split = pickle.load(open(split_name, 'rb')) # could use same split across different experiments for consistency
            except:
                if ratio_valid_test[1] > 0:
                    tr_va_te_split = {c: train_valid_test_split(cls_id_map[c], ratio_valid_test=ratio_valid_test)
                                      for c in
                                      cls}
                    print(f'splitting is {ratio_valid_test}')
                elif ratio_valid_test[1] == 0:
                    tr_va_te_split = {c: [cls_id_map[c], [], []] for c in
                                      cls}
                with open(split_name, 'wb') as f:
                    pickle.dump(tr_va_te_split, f)
        super().__init__([[graph, label], tr_va_te_split], n_cls)


class TimeIncrementalDataset(incremental_graph_trans_):
    def __init__(self, name='Elliptic-CL', n_time_tasks=20, train_ratio=0.7, args=None):
        """
        Time-based incremental dataset
        
        Args:
            name: dataset name ('Arxiv-CL' or 'elliptic')
            n_time_tasks: number of time-based tasks to create
            train_ratio: ratio of nodes used for training in each task
            args: arguments containing data paths
        """
        # Load dataset
        if name == 'Elliptic-CL':
            graph = get_elliptic_graph(args)
            label = graph.ndata['label'].view(-1, 1) 
        elif name == 'Arxiv-CL':
            data = DglNodePropPredDataset('ogbn-arxiv', root=f'{args.ori_data_path}/ogb_downloaded')
            graph, label = data[0]
            graph.ndata['label'] = label
            graph.ndata['time'] = graph.ndata['year'].view(-1)
        else:
            raise ValueError(f'Time information not available for dataset: {name}')
        
        # Process labels and remove small classes
        n_cls = max(label) + 1        
        # Create time-based task splits
        self.time_values = graph.ndata['time'].numpy()
        self.n_time_tasks = n_time_tasks
        self.train_ratio = train_ratio

        # sort by time steps
        if name == 'Elliptic-CL':
            # only use valid label
            labels_np = graph.ndata['label'].view(-1).numpy()
            valid_mask = labels_np != -1
            valid_indices = np.where(valid_mask)[0]
            sorted_indices = valid_indices[np.argsort(self.time_values[valid_indices])]
        else:
            sorted_indices = np.argsort(self.time_values)

        # Split nodes into time tasks
        task_size = len(sorted_indices) // n_time_tasks
        self.time_task_splits = {}
        
        for task_id in range(n_time_tasks):
            start_idx = task_id * task_size
            if task_id == n_time_tasks - 1:  # Last task gets remaining nodes
                end_idx = len(sorted_indices)
            else:
                end_idx = (task_id + 1) * task_size
            
            task_nodes = sorted_indices[start_idx:end_idx]
            
            # Split each task into train/test
            task_train_size = int(len(task_nodes) * train_ratio)
            task_train_ids = task_nodes[:task_train_size]
            task_test_ids = task_nodes[task_train_size:]
            
            self.time_task_splits[task_id] = [task_train_ids.tolist(), [], task_test_ids.tolist()]
            
            print(f"Task {task_id}: {len(task_train_ids)} train, {len(task_test_ids)} test nodes")
            print(f"  Time range: {self.time_values[task_nodes].min()} - {self.time_values[task_nodes].max()}")
        
        # Initialize parent class with time-based splits
        super().__init__([[graph, label], self.time_task_splits], n_cls)
    
    def get_graph(self, tasks_to_retain=[], node_ids=None):
        """
        Get subgraph for time-based tasks
        
        Args:
            tasks_to_retain: list of time task IDs to retain
            node_ids: specific node IDs [train_ids, test_ids] for a task
        
        Returns:
            subgraph: DGL subgraph
            [ids_train, ids_test]: remapped train and test node indices
        """
        return super().get_graph(tasks_to_retain, node_ids)
    
    def get_time_task_info(self, task_id):
        """Get information about a specific time task"""
        if task_id not in self.time_task_splits:
            raise ValueError(f"Task {task_id} does not exist")
        
        train_ids, _, test_ids = self.time_task_splits[task_id]
        task_nodes = np.concatenate([train_ids, test_ids])
        
        return {
            'task_id': task_id,
            'train_nodes': len(train_ids),
            'test_nodes': len(test_ids),
            'total_nodes': len(task_nodes),
            'time_range': (self.time_values[task_nodes].min(), self.time_values[task_nodes].max()),
            'train_time_range': (self.time_values[train_ids].min(), self.time_values[train_ids].max()),
            'test_time_range': (self.time_values[test_ids].min(), self.time_values[test_ids].max())
        }


def get_elliptic_graph(args):
    url = 'https://data.pyg.org/datasets/elliptic'
    raw_dir = f'{args.ori_data_path}/elliptic'
    os.makedirs(raw_dir, exist_ok=True)

    raw_file_names = [
            'elliptic_txs_features.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_classes.csv',
        ]
    for file_name in raw_file_names:
        if not os.path.exists(os.path.join(raw_dir, file_name)):
            fs.cp(f'{url}/{file_name}.zip', raw_dir, extract=True)

    feat_df = pd.read_csv(os.path.join(raw_dir, raw_file_names[0]), header=None)
    edge_df = pd.read_csv(os.path.join(raw_dir, raw_file_names[1]))
    class_df = pd.read_csv(os.path.join(raw_dir, raw_file_names[2]))

    columns = {0: 'txId', 1: 'time_step'}
    feat_df = feat_df.rename(columns=columns)

    x = torch.from_numpy(feat_df.loc[:, 2:].values).to(torch.float)

    # 0=licit,  1=illicit, -1=unknown
    mapping = {'unknown': -1, '1': 1, '2': 0}
    class_df['class'] = class_df['class'].map(mapping)
    y = torch.from_numpy(class_df['class'].values)

    mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
    edge_df['txId1'] = edge_df['txId1'].map(mapping)
    edge_df['txId2'] = edge_df['txId2'].map(mapping)
    edge_index = torch.from_numpy(edge_df.values).t().contiguous()

    time_step = torch.from_numpy(feat_df['time_step'].values)

    elliptic_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=len(feat_df))
    elliptic_graph.ndata['feat'] = x
    elliptic_graph.ndata['label'] = y
    elliptic_graph.ndata['time'] = time_step

    return elliptic_graph