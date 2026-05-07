import os
import pickle
import re
import numpy as np
import torch
from Backbones.model_factory import get_model
from dataset.utils import NodeLevelDataset
from training.utils import mkdir_if_missing, shuffle_list
from gaussian_utils import compute_centers, gaussian_task_weights, build_gaussian_stream
import importlib
import copy
import dgl
import time
import random

joint_alias = ['joint', 'Joint', 'joint_replay_all', 'jointtrain']
def get_pipeline(args):
    # choose the pipeline for the chosen setting
    # Joint training method uses a special pipeline that trains on all tasks jointly
    if args.method in joint_alias:
        return pipeline_joint_gaussian
    elif args.setting == 'tfocis':
        return pipeline_tfocis
    elif args.setting == 'tfo_bb':
        return pipeline_tfobb
    elif args.setting == 'tfo_gaussian':
        return pipeline_gaussian
    else:
        return pipeline_tfo

def data_prepare(dataset, args):
    """
    check whether the processed data exist or create new processed data
    if args.load_check is True, loading data will be tried, else, will only check the existence of the files
    """
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    # check whether the preprocessed data exist and can be loaded
    # str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    try:
        if args.load_check:
           subgraph, [train_ids, test_ids] = pickle.load(open(
                f'{args.data_path}/{args.dataset}.pkl', 'rb'))
        else:
            if f'{args.dataset}.pkl' not in os.listdir(f'{args.data_path}'):
                subgraph, [train_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{args.dataset}.pkl', 'rb'))
    except:
        # if not exist or cannot be loaded correctly, create new processed data
        print(f'preparing data')
        tasks_tr = []
        tasks_te = []
        subgraphs = []
        for task, task_cls in enumerate(args.task_seq):
            subgraph, [train_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls)
            subgraphs.append(subgraph)
            tasks_tr.append(train_ids)
            tasks_te.append(test_ids)
        with open(f'{args.data_path}/{args.dataset}.pkl', 'wb') as f:
            pickle.dump([subgraphs, [tasks_tr, tasks_te]], f)

def data_prepare_blurry(dataset, args):
    """check whether the processed data exist or create new processed data for blurry task distribution
    if args.load_check is True, loading data will be tried, else, will only check the existence of the files
    """
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    p = args.percentage
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    name = f'blurry{int(round((1.0 - p)*100))}'
    try:
        if args.load_check:
           subgraph, [train_ids, test_ids] = pickle.load(open(
                f'{args.data_path}/{args.dataset}_'+ name + '.pkl', 'rb'))
        else:
            if f'{args.data_path}/{args.dataset}_'+ name + '.pkl' not in os.listdir(f'{args.data_path}'):
                subgraph, [train_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{args.dataset}_'+ name + '.pkl', 'rb'))
    except:
        print(f'preparing data')
        subgraphs = []
        tasks_tr = []
        tasks_te = []
        tasks_tr_inds={}
        tasks_te_old = []
        for i, task_cls in enumerate(args.task_seq):
            ids_train_t = []
            ids_test_t = []
            for t in task_cls:
                ids_train_t.extend(dataset.tr_va_te_split[t][0])
                ids_test_t.extend(dataset.tr_va_te_split[t][2])
            tasks_tr_inds[i] = ids_train_t
            tasks_te_old.append(ids_test_t)
        
        # blurry task distribution
        within_task_ids = {} # retained samples in given task
        blurred_ids = []

        for t in tasks_tr_inds.keys():
            n = len(tasks_tr_inds[t])
            random.shuffle(tasks_tr_inds[t])
            within_task_ids[t] = tasks_tr_inds[t][0: int(n * p)]
            in_set_inds = set(within_task_ids[t])
            all_set_inds = set(tasks_tr_inds[t])
            out_set_inds = all_set_inds - in_set_inds # not selected samples
            out_task_inds = list(out_set_inds)
            blurred_ids.extend(out_task_inds)
            print("blurred ids len",len(blurred_ids))

        random.shuffle(blurred_ids)
        per_task_blurred = int(len(blurred_ids)/args.n_tasks)

        for t in range(args.n_tasks):
            task_tr = within_task_ids[t]
            start = t * per_task_blurred
            end = start + per_task_blurred
            print("start:",start," end:",end)
            this_out_inds = blurred_ids[start:end]
            task_tr.extend(this_out_inds)  # extend modifies the list in place
            task_te = tasks_te_old[t]
            subgraph, [train_ids, test_ids] = dataset.get_graph(node_ids=[task_tr, task_te])
            subgraphs.append(subgraph)
            tasks_tr.append(train_ids)
            tasks_te.append(test_ids)
        with open(f'{args.data_path}/{args.dataset}_'+ name + '.pkl', 'wb') as f:
            pickle.dump([subgraphs, [tasks_tr, tasks_te]], f)


def data_prepare_boundaryblurry(dataset, args):
    """check whether the processed data exist or create new processed data for boundary blurry task distribution
    This method only mixes samples at the boundary between adjacent tasks:
    - In the last K batches of task t, introduce samples from task t+1
    - In the first K batches of task t+1, retain some samples from task t
    
    The boundary region is shared: task t's last K batches = task t+1's first K batches (mixed region)
    
    if args.load_check is True, loading data will be tried, else, will only check the existence of the files
    """
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    
    # Get batch size for calculating batch boundaries
    batch_size = getattr(args, 'batch_size', 10)
    # Get number of batches to mix at boundaries
    K = getattr(args, 'blurry_batch_count', 2)
    # Get mixing ratio (proportion of samples from adjacent task in boundary batches)
    mix_ratio = getattr(args, 'boundary_mix_ratio', 0.5)
    
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task)) for i in range(0, args.n_cls-1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks = len(args.task_seq)
    
    name = f'boundaryblurry_K{K}_ratio{int(mix_ratio*100)}'
    
    try:
        if args.load_check:
           subgraph, [train_ids, test_ids] = pickle.load(open(
                f'{args.data_path}/{args.dataset}_'+ name + '.pkl', 'rb'))
        else:
            if f'{args.data_path}/{args.dataset}_'+ name + '.pkl' not in os.listdir(f'{args.data_path}'):
                subgraph, [train_ids, test_ids] = pickle.load(open(
                    f'{args.data_path}/{args.dataset}_'+ name + '.pkl', 'rb'))
    except:
        print(f'preparing boundary blurry data with K={K}, mix_ratio={mix_ratio}, batch_size={batch_size}')
        subgraphs = []
        tasks_tr = []
        tasks_te = []
        tasks_tr_inds = {}
        tasks_te_old = []
        
        # Step 1: Collect original task samples
        for i, task_cls in enumerate(args.task_seq):
            ids_train_t = []
            ids_test_t = []
            for t in task_cls:
                ids_train_t.extend(dataset.tr_va_te_split[t][0])
                ids_test_t.extend(dataset.tr_va_te_split[t][2])
            tasks_tr_inds[i] = ids_train_t
            tasks_te_old.append(ids_test_t)
        
        # Step 2: Shuffle each task's samples (to randomize batch composition)
        for t in tasks_tr_inds.keys():
            random.shuffle(tasks_tr_inds[t])
        
        # Step 3: Process boundary mixing for adjacent tasks
        # Strategy: For each boundary between task t and t+1:
        #   1. Extract last K*batch_size samples from task t (boundary part of t) - these will be removed from task t
        #   2. Extract first K*batch_size samples from task t+1 (boundary part of t+1) - these will be removed from task t+1
        #   3. Mix them: create a mixed boundary with mix_ratio from t+1 and (1-mix_ratio) from t
        #   4. Task t gets: [pure t samples (excluding boundary)] + [mixed boundary with t+1]
        #   5. Task t+1 gets: [mixed boundary with t] + [pure t+1 samples (excluding boundary)]
        #   6. For middle tasks: [mixed boundary with t-1] + [pure samples] + [mixed boundary with t+1]
        
        final_tasks_tr = {}
        boundary_size_samples = K * batch_size
        
        # First pass: extract boundary samples and create mixed boundaries
        # mixed_boundaries[t] = mixed boundary between task t and t+1
        mixed_boundaries = {}
        
        # Initialize main_parts with all original samples
        # As we process boundaries, we'll remove the boundary parts from main_parts
        main_parts = {t: tasks_tr_inds[t].copy() for t in range(args.n_tasks)}
        
        for t in range(args.n_tasks - 1):  # Process all boundaries
            # Get current state of task samples (may have been modified in previous iterations)
            task_samples = main_parts[t].copy()
            next_task_samples = main_parts[t + 1].copy()
            
            # Extract boundary from task t (last K batches) - these will be removed from task t
            boundary_size_t = min(boundary_size_samples, len(task_samples))
            boundary_t = task_samples[-boundary_size_t:] if boundary_size_t > 0 else []
            main_t = task_samples[:-boundary_size_t] if boundary_size_t < len(task_samples) else []
            
            # Extract boundary from task t+1 (first K batches) - these will be removed from task t+1
            boundary_size_next = min(boundary_size_samples, len(next_task_samples))
            boundary_next = next_task_samples[:boundary_size_next] if boundary_size_next > 0 else []
            main_next = next_task_samples[boundary_size_next:] if boundary_size_next < len(next_task_samples) else []
            
            # Update main_parts: remove the boundary parts
            main_parts[t] = main_t
            main_parts[t + 1] = main_next
            
            # Mix boundaries
            # Calculate number of samples from each task in the mixed boundary
            # The mixed boundary should have approximately mix_ratio from next task
            total_boundary_size = boundary_size_samples  # Target size for mixed boundary
            n_from_next = int(total_boundary_size * mix_ratio)
            n_from_t = total_boundary_size - n_from_next
            
            # Adjust based on available samples
            n_from_t = min(n_from_t, len(boundary_t))
            n_from_next = min(n_from_next, len(boundary_next))
            
            # Select and mix
            random.shuffle(boundary_t)
            random.shuffle(boundary_next)
            mixed_boundary = boundary_t[:n_from_t] + boundary_next[:n_from_next]
            random.shuffle(mixed_boundary)
            
            mixed_boundaries[t] = mixed_boundary
            
            print(f"Boundary {t}-{t+1}: mixed {len(mixed_boundary)} samples "
                  f"({n_from_t} from task {t}, {n_from_next} from task {t+1}), "
                  f"task {t} main after removal: {len(main_t)}, task {t+1} main after removal: {len(main_next)}")
        
        # Second pass: construct final training sets
        # Each task should have: [mixed boundary with previous] + [pure samples] + [mixed boundary with next]
        # The pure samples are from main_parts[t], which already has boundary parts removed
        for t in range(args.n_tasks):
            parts = []
            
            # Add mixed boundary with previous task (if exists)
            # This is the boundary between task t-1 and t
            if t > 0 and (t - 1) in mixed_boundaries:
                parts.append(mixed_boundaries[t - 1])
            
            # Add pure samples (main part, excluding boundaries that were extracted)
            # main_parts[t] already has the boundary parts removed
            parts.append(main_parts[t])
            
            # Add mixed boundary with next task (if exists)
            # This is the boundary between task t and t+1
            if t < args.n_tasks - 1 and t in mixed_boundaries:
                parts.append(mixed_boundaries[t])
            
            # Flatten list of lists
            final_tasks_tr[t] = sum(parts, [])
            
            # Verify no duplicates
            unique_samples = set(final_tasks_tr[t])
            if len(final_tasks_tr[t]) != len(unique_samples):
                print(f"Warning: Task {t} has {len(final_tasks_tr[t]) - len(unique_samples)} duplicate samples!")
                # Remove duplicates while preserving order
                seen = set()
                final_tasks_tr[t] = [x for x in final_tasks_tr[t] if not (x in seen or seen.add(x))]
            
            print(f"Task {t}: {len(final_tasks_tr[t])} total samples "
                  f"(pure: {len(main_parts[t])}, "
                  f"boundary with prev: {len(mixed_boundaries.get(t-1, []))}, "
                  f"boundary with next: {len(mixed_boundaries.get(t, []))})")
        
        for t in range(args.n_tasks):
            task_tr = final_tasks_tr[t]
            task_te = tasks_te_old[t]
            subgraph, [train_ids, test_ids] = dataset.get_graph(node_ids=[task_tr, task_te])
            subgraphs.append(subgraph)
            tasks_tr.append(train_ids)
            tasks_te.append(test_ids)
        
        with open(f'{args.data_path}/{args.dataset}_'+ name + '.pkl', 'wb') as f:
            pickle.dump([subgraphs, [tasks_tr, tasks_te]], f)


def data_prepare_tem(dataset, args):
    """
    check whether the processed data exist or create new processed data
    if args.load_check is True, loading data will be tried, else, will only check the existence of the files
    """
    if args.cuda:
        torch.cuda.set_device(args.gpu)
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    args.n_tasks = dataset.n_time_tasks
    pkl_path = f'{args.data_path}/{args.dataset}_tem{args.n_time_tasks}.pkl'
    # check whether the preprocessed data exist and can be loaded
    # str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    try:
        if args.load_check:
           subgraph, [train_ids, test_ids] = pickle.load(open(
                pkl_path, 'rb'))
        else:
            if f'{args.dataset}_tem{args.n_time_tasks}.pkl' not in os.listdir(f'{args.data_path}'):
                subgraph, [train_ids, test_ids] = pickle.load(open(
                    pkl_path, 'rb'))
    except:
        # if not exist or cannot be loaded correctly, create new processed data
        print(f'preparing data')
        tasks_tr = []
        tasks_te = []
        subgraphs = []
        for task_id in range(dataset.n_time_tasks):
            # Get task info
            # task_info = dataset.get_time_task_info(task_id)
            # print(f"Task info: {task_info}")
            
            subgraph, [train_ids, test_ids] = dataset.get_graph(tasks_to_retain=[task_id])
            subgraphs.append(subgraph)
            tasks_tr.append(train_ids)
            tasks_te.append(test_ids)
        with open(pkl_path, 'wb') as f:
            pickle.dump([subgraphs, [tasks_tr, tasks_te]], f)
    

def eval_tasks(model, continuum, tasks_te, cur_t, args):
    model.eval()
    result = [] # acc for each task
    total_size = 0
    total_pred = 0
    current_result = [] # result til current task
    current_avg_acc = 0 # avg acc til current task
    for i, task_te in enumerate(tasks_te):
        t = i
        subgraph = continuum.graphs[t]
        if args.cuda:
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        with torch.no_grad():
            output, _ = model(subgraph, features)
            logits = output[task_te]
            labels = labels[task_te]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            accuracy =  correct.item() * 1.0 / len(labels)
            result.append(accuracy)
            total_size += len(labels)
            total_pred += correct.item()
        
        if t == cur_t:
            current_result = [res for res in result]
            current_avg_acc = total_pred * 1.0 / total_size
        
    # torch.save((model.state_dict(), current_result, current_avg_acc), args.method + '.pt')
    
    return result, total_pred * 1.0 / total_size, current_result, current_avg_acc


def eval_tasks_cis(model, continuum, tasks_te, cur_t, args):
    model.eval()
    cls_seen_so_far = set()
    result = [] # acc for each task
    total_size = 0
    total_pred = 0
    current_result = [] # result til current task
    current_avg_acc = 0 # avg acc til current task
    if args.setting in ['tfo_bb', 'tfo_gaussian']:
        subgraphs = continuum
    else:
        subgraphs = continuum.graphs
    for i, task_te in enumerate(tasks_te):
        t = i
        subgraph = subgraphs[t]
        if args.cuda:
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        cls_seen_so_far.update(labels[task_te].unique()) # update the classes (in test set) seen so far
        offset1, offset2 = 0, max(cls_seen_so_far)+1
        if offset2 % 2 != 0:
            offset2 += 1
        with torch.no_grad():
            output, _ = model(subgraph, features)
            logits = output[task_te][:, offset1:offset2]
            labels = labels[task_te]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            accuracy =  correct.item() * 1.0 / len(labels)
            result.append(accuracy)
            total_size += len(labels)
            total_pred += correct.item()
        
        if t == cur_t:
            current_result = [res for res in result]
            current_avg_acc = total_pred * 1.0 / total_size
            print(f'acc till {t}: {current_avg_acc}')
        
    # torch.save((model.state_dict(), current_result, current_avg_acc), args.method + '.pt')
    
    return result, total_pred * 1.0 / total_size, current_result, current_avg_acc

def eval_tasks_batch(model, continuum, tasks_te, cur_t, args):
    model.eval()
    result = []
    total_size = 0
    total_pred = 0
    current_result = []
    current_avg_acc = 0

    if args.setting == 'tfo_bb':
        subgraphs = continuum
    else:
        subgraphs = continuum.graphs

    sampler = dgl.dataloading.NeighborSampler(
        args.n_nbs_sample
    ) if args.sample_nbs else \
    dgl.dataloading.MultiLayerFullNeighborSampler(
        model.n_layers
    )
    bs = args.batch_size

    for t, task_te in enumerate(tasks_te):
        g = subgraphs[t]
        if args.cuda:
            g = g.to(device='cuda:{}'.format(args.gpu))
        labels = g.dstdata['label'].squeeze()

        correct_t = 0
        total_t = 0

        with torch.no_grad():
            for start in range(0, len(task_te), bs):
                end = start + bs
                ids_bt = task_te[start:end]
                _, _, blocks = sampler.sample_blocks(g, ids_bt)
                feats = blocks[0].srcdata['feat']
                batch_labels = labels[ids_bt]

                output, _ = model.forward_batch(blocks, feats)
                # logits = output[:, offset1:offset2]

                _, pred = torch.max(output, dim=1)
                correct_t += (pred == batch_labels).sum().item()
                total_t += len(batch_labels)

        acc = correct_t / total_t
        result.append(acc)

        total_pred += correct_t
        total_size += total_t

        if t == cur_t:
            current_result = [res for res in result]
            current_avg_acc = total_pred / total_size
            # print(f'acc till t: {current_avg_acc}')

    return result, total_pred / total_size, current_result, current_avg_acc


def eval_tasks_cis_batch(model, continuum, tasks_te, cur_t, args):
    model.eval()
    cls_seen_so_far = set()
    result = []
    total_size = 0
    total_pred = 0
    current_result = []
    current_avg_acc = 0

    if args.setting in ['tfo_bb', 'tfo_gaussian']:
        subgraphs = continuum
    else:
        subgraphs = continuum.graphs

    sampler = dgl.dataloading.NeighborSampler(
        args.n_nbs_sample
    ) if args.sample_nbs else \
    dgl.dataloading.MultiLayerFullNeighborSampler(
        model.n_layers
    )
    bs = args.batch_size

    for t, task_te in enumerate(tasks_te):
        g = subgraphs[t]
        if args.cuda:
            g = g.to(device='cuda:{}'.format(args.gpu))
        labels = g.dstdata['label'].squeeze()

        cls_seen_so_far.update(labels[task_te].unique())
        offset1, offset2 = 0, max(cls_seen_so_far) + 1
        if offset2 % 2 != 0:
            offset2 += 1

        correct_t = 0
        total_t = 0

        with torch.no_grad():
            for start in range(0, len(task_te), bs):
                end = start + bs
                ids_bt = task_te[start:end]
                _, _, blocks = sampler.sample_blocks(g, ids_bt)
                feats = blocks[0].srcdata['feat']
                batch_labels = labels[ids_bt]

                output, _ = model.forward_batch(blocks, feats)
                logits = output[:, offset1:offset2]

                _, pred = torch.max(logits, dim=1)
                correct_t += (pred == batch_labels).sum().item()
                total_t += len(batch_labels)

        acc = correct_t / total_t
        result.append(acc)

        total_pred += correct_t
        total_size += total_t

        if t == cur_t:
            current_result = [res for res in result]
            current_avg_acc = total_pred / total_size
            print(f'acc till {t}: {current_avg_acc}')

    return result, total_pred / total_size, current_result, current_avg_acc


def pipeline_tfo(dataset, continuum, tasks_te, args):
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    epochs = args.epochs
    model = get_model(dataset, args)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    # Pass dataset to GSS, AGEM, ER, and DMSG models for accessing full graph
    if args.method in ['gss', 'agem', 'er', 'ergnn', 'dmsg']:
        life_model_ins = life_model.NET(model, args, dataset=dataset)
    else:
        life_model_ins = life_model.NET(model, args)
    
    # if args.dataset == 'Reddit-CL':
    #     eval = eval_tasks_batch
    # else:
    #     eval = eval_tasks
    if args.eval_batch:
        eval = eval_tasks_batch
    else:
        eval = eval_tasks
    result_list = [] 
    current_result_list = [] # result til current task
    avg_acc_list = [] # avg accuracy on task seen so far
    current_avg_acc_list = [] # avg acc til current task
    task_list = [] # task list
    current_task = 0
    time_start = time.time()

    for (i, (subgraph, t, ids_batch)) in enumerate(continuum):
        # only training on data of one task each time
        if t > args.n_tasks:
            break
        if(((i % args.log_every) == 0) or (t != current_task)):
        # if t != current_task: # eval when task changes
            res_per_t, avg_acc, current_res_per_t, current_avg_acc = eval(model, continuum, tasks_te, current_task, args)
            result_list.append(res_per_t)
            avg_acc_list.append(avg_acc)
            current_result_list.append(current_res_per_t)
            current_avg_acc_list.append(current_avg_acc)
            task_list.append(current_task)
            
            current_task = t
        
        if args.cuda:
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        torch.cuda.empty_cache()
        
        for epoch in range(epochs):
            # Train classification model
            life_model_ins.observe(args, subgraph, features, labels, ids_batch)
    
    res_per_t, avg_acc, current_res_per_t, current_avg_acc = eval(model, continuum, tasks_te, args.n_tasks-1, args) # test after training
    result_list.append(res_per_t)
    avg_acc_list.append(avg_acc)
    current_result_list.append(current_res_per_t)
    current_avg_acc_list.append(current_avg_acc)
    task_list.append(current_task)

    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_save_path}/{args.method}_{args.dataset}_{args.setting}_{args.seed}.pt')

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_list), torch.Tensor(avg_acc_list), current_result_list, torch.Tensor(current_avg_acc_list), torch.Tensor(task_list), time_spent

def pipeline_tfocis(dataset, continuum, tasks_te, args):
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    epochs = args.epochs
    model = get_model(dataset, args)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    # Pass dataset to GSS, AGEM, ER, and DMSG models for accessing full graph
    if args.method in ['gss', 'agem', 'er', 'ergnn', 'dmsg']:
        life_model_ins = life_model.NET(model, args, dataset=dataset)
    else:
        life_model_ins = life_model.NET(model, args)
    
    if args.eval_batch:
        eval = eval_tasks_cis_batch
    else:
        eval = eval_tasks_cis

    result_list = [] 
    current_result_list = [] # result til current task
    avg_acc_list = [] # avg accuracy on task seen so far
    current_avg_acc_list = [] # avg acc til current task
    task_list = [] # task list
    current_task = 0
    time_start = time.time()

    for (i, (subgraph, t, ids_batch)) in enumerate(continuum):
        # only training on data of one task each time
        if t > args.n_tasks:
            break
        if(((i % args.log_every) == 0) or (t != current_task)):
        # if t != current_task: # eval when task changes
            res_per_t, avg_acc, current_res_per_t, current_avg_acc = eval(model, continuum, tasks_te, current_task, args)
            result_list.append(res_per_t)
            avg_acc_list.append(avg_acc)
            current_result_list.append(current_res_per_t)
            current_avg_acc_list.append(current_avg_acc)
            task_list.append(current_task)
            
            current_task = t
        
        if args.cuda:
            subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
        torch.cuda.empty_cache()
        
        for epoch in range(epochs):
            # Train classification model
            life_model_ins.observe_cis(args, subgraph, features, labels, ids_batch)
    
    res_per_t, avg_acc, current_res_per_t, current_avg_acc = eval(model, continuum, tasks_te, args.n_tasks-1, args) # test after training
    result_list.append(res_per_t)
    avg_acc_list.append(avg_acc)
    current_result_list.append(current_res_per_t)
    current_avg_acc_list.append(current_avg_acc)
    task_list.append(current_task)

    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_save_path}/{args.method}_{args.dataset}_{args.setting}_{args.seed}.pt')

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_list), torch.Tensor(avg_acc_list), current_result_list, torch.Tensor(current_avg_acc_list), torch.Tensor(task_list), time_spent

def pipeline_tfobb(dataset, data, tasks_te, args):
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    epochs = args.epochs
    batch_size = args.batch_size
    # data = [subgraphs, [tasks_tr, tasks_te]]
    subgraphs = data[0]
    tasks_tr = data[1][0]
    # tasks_te = data[1][1]
    model = get_model(dataset, args)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    life_model = importlib.import_module(f'Baselines.{args.method}_model')

    if args.method in ['gss', 'agem', 'er', 'ergnn', 'dmsg']:
        life_model_ins = life_model.NET(model, args, dataset=dataset)
    else:
        life_model_ins = life_model.NET(model, args)

    if args.eval_batch:
        eval = eval_tasks_cis_batch
    else:
        eval = eval_tasks_cis
    result_list = [] 
    current_result_list = [] # result til current task
    avg_acc_list = [] # avg accuracy on task seen so far
    current_avg_acc_list = [] # avg acc til current task
    task_list = [] # task list
    current_task = 0
    time_start = time.time()

    bnc = 0
    for t, subgraph in enumerate(subgraphs):
        if t >= args.n_tasks:
            break
        train_node_ids = tasks_tr[t]
        if len(train_node_ids) == 0:
            continue

        # Iterate batches sequentially according to `train_node_ids` order
        for start in range(0, len(train_node_ids), batch_size):
            end = start + batch_size
            ids_batch_list = train_node_ids[start:end]
            ids_batch = torch.LongTensor(ids_batch_list)

            # Evaluation schedule (same logic as other pipelines, but using `subgraphs`)
            if ((bnc % args.log_every) == 0) or (t != current_task):
                res_per_t, avg_acc, current_res_per_t, current_avg_acc = eval(model, subgraphs, tasks_te, current_task, args)
                result_list.append(res_per_t)
                avg_acc_list.append(avg_acc)
                current_result_list.append(current_res_per_t)
                current_avg_acc_list.append(current_avg_acc)
                task_list.append(current_task)
                
                current_task = t

            if args.cuda:
                subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
            features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
            torch.cuda.empty_cache()

            for epoch in range(epochs):
                # Train classification model
                life_model_ins.observe_cis(args, subgraph, features, labels, ids_batch)

            bnc += 1

    res_per_t, avg_acc, current_res_per_t, current_avg_acc = eval(model, subgraphs, tasks_te, args.n_tasks-1, args) # test after training
    result_list.append(res_per_t)
    avg_acc_list.append(avg_acc)
    current_result_list.append(current_res_per_t)
    current_avg_acc_list.append(current_avg_acc)
    task_list.append(current_task)

    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_save_path}/{args.method}_{args.dataset}_{args.setting}_{args.seed}.pt')

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_list), torch.Tensor(avg_acc_list), current_result_list, torch.Tensor(current_avg_acc_list), torch.Tensor(task_list), time_spent


def data_prepare_gaussian(dataset, args):
    """
    Build and cache the data for the Softmax-Gaussian blurry CGL setting.

    Pkl layout
    ----------
    data[0]  list of per-task eval subgraphs
    data[1]  [tasks_tr_local, tasks_te_local]   (data[1][1] consumed by main.py)
    data[2]  merged training subgraph
    data[3]  orig_to_local dict  {orig_graph_node_id -> local_idx_in_merged}
    data[4]  stream  list of (batch_orig_ids, batch_task_labels, weights)
    """
    sigma      = getattr(args, 'gaussian_sigma', 10.0)
    batch_size = args.batch_size
    replace    = getattr(args, 'replace', getattr(args, 'gaussian_replace', False))
    seed       = getattr(args, 'seed', 0)

    if args.cuda:
        torch.cuda.set_device(args.gpu)

    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    cls = [list(range(i, i + args.n_cls_per_task))
           for i in range(0, args.n_cls - 1, args.n_cls_per_task)]
    args.task_seq = cls
    args.n_tasks  = len(args.task_seq)

    # Important: cache must distinguish with/without replacement streams.
    # Otherwise toggling --replace would silently reuse old cached data.
    name = f'gaussian_sigma{sigma}_bs{batch_size}_rep{int(bool(replace))}'
    pkl_path = f'{args.data_path}/{args.dataset}_{name}.pkl'

    try:
        if args.load_check:
            pickle.load(open(pkl_path, 'rb'))
        else:
            if not os.path.exists(pkl_path):
                raise FileNotFoundError
        return   # cache hit
    except Exception:
        # pass

        print(f'[gaussian] preparing data  sigma={sigma}  batch_size={batch_size}')

        # Per-task original node ids
        tasks_tr_orig = {}
        tasks_te_orig = []
        for i, task_cls in enumerate(args.task_seq):
            tr_ids, te_ids = [], []
            for c in task_cls:
                tr_ids.extend(dataset.tr_va_te_split[c][0])
                te_ids.extend(dataset.tr_va_te_split[c][2])
            tasks_tr_orig[i] = tr_ids
            tasks_te_orig.append(te_ids)

        eval_subgraphs = []
        tasks_tr_local = []
        tasks_te_local = []
        for t in range(args.n_tasks):
            subg, [tr_loc, te_loc] = dataset.get_graph(
                node_ids=[tasks_tr_orig[t], tasks_te_orig[t]]
            )
            eval_subgraphs.append(subg)
            tasks_tr_local.append(tr_loc)
            tasks_te_local.append(te_loc)

        all_tr = sum(tasks_tr_orig.values(), [])
        all_te = sum(tasks_te_orig, [])
        merged_subgraph, _ = dataset.get_graph(node_ids=[all_tr, all_te])

        orig_ids_t  = merged_subgraph.ndata['_ID'].cpu()
        orig_to_local = {int(orig_ids_t[i]): i for i in range(orig_ids_t.shape[0])}

        # construct data stream
        task_node_ids = [tasks_tr_orig[t] for t in range(args.n_tasks)]
        result = build_gaussian_stream(
            task_node_ids, batch_size, sigma, seed=seed, replace=replace,
        )
        stream = result[0]

        with open(pkl_path, 'wb') as f:
            pickle.dump([eval_subgraphs,
                        [tasks_tr_local, tasks_te_local],
                        merged_subgraph,
                        orig_to_local,
                        stream], f)
        # print(f'[gaussian] saved to {pkl_path}')


def pipeline_gaussian(dataset, data, tasks_te, args):
    """
    Training pipeline for the Softmax-Gaussian blurry CGL setting (class-IL).

    data layout (loaded from pkl built by data_prepare_gaussian):
      data[0]  eval_subgraphs   list[DGL graph]
      data[1]  [tasks_tr_local, tasks_te_local]
      data[2]  merged_subgraph  DGL graph (all tasks combined)
      data[3]  orig_to_local    dict
      data[4]  stream           list of (batch_orig_ids, batch_task_labels, weights)
    """
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls
    epochs = args.epochs

    eval_subgraphs = data[0]
    merged_subgraph = data[2]
    orig_to_local = data[3]
    stream = data[4]

    model = get_model(dataset, args)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        merged_subgraph = merged_subgraph.to(device='cuda:{}'.format(args.gpu))

    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    if args.method in ['gss', 'agem', 'er', 'ergnn', 'dmsg']:
        life_model_ins = life_model.NET(model, args, dataset=dataset)
    else:
        life_model_ins = life_model.NET(model, args)

    if args.eval_batch:
        eval = eval_tasks_cis_batch
    else:
        eval = eval_tasks_cis

    features = merged_subgraph.srcdata['feat']
    labels = merged_subgraph.dstdata['label'].squeeze()

    result_list = []
    avg_acc_list = []
    current_result_list = []
    current_avg_acc_list = []
    task_list = []
    current_task = 0  
    time_start = time.time()

    for b, (batch_orig_ids, _, weights) in enumerate(stream):
        if b % args.log_every == 0:
            res_per_t, avg_acc, cur_res, cur_acc = eval(
                model, eval_subgraphs, tasks_te, current_task, args
            )
            result_list.append(res_per_t)
            avg_acc_list.append(avg_acc)
            current_result_list.append(cur_res)
            current_avg_acc_list.append(cur_acc)
            task_list.append(current_task)

        current_task = int(np.argmax(weights))

        local_ids = torch.LongTensor([orig_to_local[nid] for nid in batch_orig_ids])
        if args.cuda:
            local_ids = local_ids.to(device='cuda:{}'.format(args.gpu))

        for _ in range(epochs):
            life_model_ins.observe_cis(args, merged_subgraph, features, labels, local_ids)

    res_per_t, avg_acc, cur_res, cur_acc = eval(
        model, eval_subgraphs, tasks_te, args.n_tasks - 1, args
    )
    result_list.append(res_per_t)
    avg_acc_list.append(avg_acc)
    current_result_list.append(cur_res)
    current_avg_acc_list.append(cur_acc)
    task_list.append(current_task)

    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_save_path}/{args.method}_{args.dataset}_{args.setting}_{args.seed}.pt')

    time_spent = time.time() - time_start

    return (torch.Tensor(result_list), torch.Tensor(avg_acc_list),
            current_result_list, torch.Tensor(current_avg_acc_list),
            torch.Tensor(task_list), time_spent)


def pipeline_joint_gaussian(dataset, data, tasks_te, args):
    """
    Upper bound for the Gaussian setting (Static Joint oracle).
    (1) Offline training once on ALL task training nodes
    (2) Walk through the Gaussian stream *without* any further training.
        Evaluate at the same `log_every` checkpoints as pipeline_gaussian so
        that the returned result_list / avg_acc_list are compatible with
        tf_metrics (A_AUC, AF_S).

    data layout (same pkl as pipeline_gaussian):
      data[0]  eval_subgraphs   list[DGL graph]
      data[1]  [tasks_tr_local, tasks_te_local]
      data[2]  merged_subgraph  DGL graph (all tasks combined)
      data[3]  orig_to_local    dict {orig_node_id -> local_idx_in_merged}
      data[4]  stream           list of (batch_orig_ids, batch_task_labels, weights)
    """
    args.d_data, args.n_cls = dataset.d_data, dataset.n_cls

    eval_subgraphs  = data[0]
    merged_subgraph = data[2]
    orig_to_local   = data[3]
    stream          = data[4]
    batch_size      = args.batch_size

    model = get_model(dataset, args)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        merged_subgraph = merged_subgraph.to(device='cuda:{}'.format(args.gpu))

    bare_model = importlib.import_module('Baselines.bare_model')
    trainer = bare_model.NET(model, args)

    if args.eval_batch:
        eval = eval_tasks_cis_batch
    else:
        eval = eval_tasks_cis

    features = merged_subgraph.srcdata['feat']
    labels   = merged_subgraph.dstdata['label'].squeeze()

    batch_train = 2000
    all_train_orig = set()
    for batch_orig_ids, _, _ in stream:
        all_train_orig.update(batch_orig_ids)
    all_train_local = np.array([orig_to_local[nid] for nid in all_train_orig])

    rng = np.random.default_rng(getattr(args, 'seed', 0))
    print(f'[joint_gaussian] joint training on {len(all_train_local)} nodes '
          f'for {args.epochs} epoch(s)...')
    time_start = time.time()

    for epoch in range(args.epochs):
        perm = rng.permutation(len(all_train_local))
        shuffled = all_train_local[perm]
        for start in range(0, len(shuffled), batch_train):
            ids_batch = torch.LongTensor(
                shuffled[start:start + batch_train].tolist()
            )
            if args.cuda:
                ids_batch = ids_batch.to(device='cuda:{}'.format(args.gpu))
            trainer.observe_cis(args, merged_subgraph, features, labels, ids_batch)

    print(f'[joint_gaussian] joint training done in {time.time() - time_start:.1f}s')

    # Eval Only
    result_list          = []
    avg_acc_list         = []
    current_result_list  = []
    current_avg_acc_list = []
    task_list            = []
    current_task = 0

    for b, (_, _, weights) in enumerate(stream):
        if b % args.log_every == 0:
            res_per_t, avg_acc, cur_res, cur_acc = eval(
                model, eval_subgraphs, tasks_te, current_task, args
            )
            result_list.append(res_per_t)
            avg_acc_list.append(avg_acc)
            current_result_list.append(cur_res)
            current_avg_acc_list.append(cur_acc)
            task_list.append(current_task)

        current_task = int(np.argmax(weights))

    # final snapshot (mirrors pipeline_gaussian)
    res_per_t, avg_acc, cur_res, cur_acc = eval(
        model, eval_subgraphs, tasks_te, args.n_tasks - 1, args
    )
    result_list.append(res_per_t)
    avg_acc_list.append(avg_acc)
    current_result_list.append(cur_res)
    current_avg_acc_list.append(cur_acc)
    task_list.append(current_task)

    if args.save_model:
        torch.save(model.state_dict(), f'{args.model_save_path}/{args.method}_{args.dataset}_{args.setting}_{args.seed}.pt')

    time_spent = time.time() - time_start

    return (torch.Tensor(result_list), torch.Tensor(avg_acc_list),
            current_result_list, torch.Tensor(current_avg_acc_list),
            torch.Tensor(task_list), time_spent)