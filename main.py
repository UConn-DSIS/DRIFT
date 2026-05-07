import os
import pickle
from re import sub
import traceback
import argparse
import sys
from distutils.util import strtobool
from pipeline import data_prepare, data_prepare_blurry, data_prepare_boundaryblurry, data_prepare_tem, data_prepare_gaussian, get_pipeline
from training.utils import set_seed, mkdir_if_missing, remove_illegal_characters,str2dict, assign_hyp_param
from Backbones.model_factory import get_model
from dataset.utils import NodeLevelDataset, Continuum, TimeIncrementalDataset, TimeContinuum
from metrics import confusion_matrix, tf_metrics
import torch


def _setup_time_streaming(args):
    """Setup time incremental learning dataset and continuum"""
    if args.dataset not in ['Elliptic-CL', 'Arxiv-CL']:
        raise ValueError(f"Time information not available for dataset: {args.dataset}")
    
    dataset = TimeIncrementalDataset(args.dataset, n_time_tasks=args.n_time_tasks, args=args)
    data_prepare_tem(dataset, args)
    
    data_path = f'{args.data_path}/{args.dataset}_tem{args.n_time_tasks}.pkl'
    data = pickle.load(open(data_path, 'rb'))
    continuum = TimeContinuum(data, args)
    
    return dataset, continuum, data


def _setup_standard(args):
    """Setup standard incremental learning dataset and continuum"""
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    
    if args.setting == 'tfo_blurry':
        data_prepare_blurry(dataset, args)
        blurry_suffix = int(round((1.0 - args.percentage) * 100))
        data_path = f'{args.data_path}/{args.dataset}_blurry{blurry_suffix}.pkl'
    elif args.setting == 'tfo_bb':
        data_prepare_boundaryblurry(dataset, args)
        K = getattr(args, 'blurry_batch_count', 2)
        mix_ratio = getattr(args, 'boundary_mix_ratio', 0.5)
        data_path = f'{args.data_path}/{args.dataset}_boundaryblurry_K{K}_ratio{int(mix_ratio*100)}.pkl'
    elif args.setting == 'tfo_gaussian':
        data_prepare_gaussian(dataset, args)
        sigma = args.gaussian_sigma
        data_path = (
            f'{args.data_path}/{args.dataset}_gaussian_sigma{sigma}_'
            f'bs{args.batch_size}_rep{int(bool(args.replace))}.pkl'
        )
    else:
        data_prepare(dataset, args)
        data_path = f'{args.data_path}/{args.dataset}.pkl'

    data = pickle.load(open(data_path, 'rb'))
    if args.setting in ('tfo_bb', 'tfo_gaussian') or args.method == 'joint':
        continuum = data
    else:
        continuum = Continuum(data, args)
    
    return dataset, continuum, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFOCGL')
    parser.add_argument("--dataset", type=str, default='CoraFull-CL', help='Reddit-CL, Arxiv-CL, CoraFull-CL, RomanEmpire-CL',)
    parser.add_argument("--gpu", type=int, default=0, help="which device to use.")
    parser.add_argument('--cuda', type=str, default='yes', help='Use GPU?')
    parser.add_argument("--seed", type=int, default=1, help="seed for exp")
    parser.add_argument("--epochs", type=int, default=1, help="number of training epochs, default = 1")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")
    parser.add_argument('--backbone', type=str, default='GCN', help="backbone GNN, [GAT, GCN, GIN]")
    parser.add_argument('--method', type=str,
                        choices=["bare", 'agem', 'mas', 'joint', 'gss', 'er', 'tfmas', 'ssm', 'dmsg', 'sem'], default="bare",
                        help="baseline continual learning method")
    parser.add_argument('--setting', type=str, default='tfocis', help="setting [tfo, tfocis, tfo_bb, tfo_gaussian]")
    parser.add_argument('--time_streaming', type=strtobool, default=False, help="whether to load time incremental graph")
    # parameters for continual learning settings
    parser.add_argument('--tasks_to_preserve', type=int, default=1,
                        help='number of tasks to preserve')
    parser.add_argument('--share-labels', type=strtobool, default=False,
                        help='task-IL specific, whether to share output label space for different tasks')
    parser.add_argument('--inter-task-edges', type=strtobool, default=True,
                        help='whether to keep the edges connecting nodes from different tasks')
    parser.add_argument('--classifier-increase', type=strtobool, default=True,
                        help='(deprecated) class-IL specific, whether to enlarge the label space with the coming of new classes, unrealistic to be set as False')

    # data parameters
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--samples_per_task', type=int, default='0',
                        help='training samples per task (all if 0)')
    parser.add_argument('--percentage', type=float, default='0.9',
                        help='percentage of samples to maintain in a given task')
    parser.add_argument('--blurry_batch_count', type=int, default=5,
                        help='number of batches to mix at task boundaries (for boundary blurry setting)')
    parser.add_argument('--boundary_mix_ratio', type=float, default=0.5,
                        help='ratio of samples from adjacent task in boundary batches (for boundary blurry setting)')
    parser.add_argument('--gaussian_sigma', type=float, default=10.0,
                        help='Gaussian width in batch units for tfo_gaussian setting (larger = more task overlap)')
    parser.add_argument('--replace', type=strtobool, default=False,
                        help='whether to use with-replacement sampling in tfo_gaussian (default: without-replacement)')

    # extra parameters
    parser.add_argument('--refresh_data', type=strtobool, default=False, help='whether to load existing splitting or regenerate')
    parser.add_argument('--d_data', default=None, help='will be assigned during running')
    parser.add_argument('--n_cls', default=None, help='will be assigned during running')
    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.2, 0.2], help='ratio of nodes used for valid and test')
    parser.add_argument('--transductive', type=strtobool, default=True, help='using transductive or inductive')
    parser.add_argument('--default_split', type=strtobool, default=False, help='whether to  use the data split provided by the dataset')
    parser.add_argument('--task_seq', default=[])
    parser.add_argument('--n-task', default=0, help='will be assigned during running')
    parser.add_argument('--n_time_tasks', type=int, default=20, help='number of time-based tasks')
    parser.add_argument('--n_cls_per_task', default=2, help='how many classes does each task  contain')
    parser.add_argument('--GAT-args',
                        default={'num_layers': 1, 'num_hidden': 32, 'heads': 8, 'out_heads': 1, 'feat_drop': .6,
                                 'attn_drop': .6, 'negative_slope': 0.2, 'residual': False})
    parser.add_argument('--GCN-args', default={'h_dims': [256], 'dropout': 0.0, 'batch_norm': False})
    parser.add_argument('--GIN-args', default={'h_dims': [256], 'dropout': 0.0})
    parser.add_argument('--ergnn_args', type=str2dict, default={'budget': [100,1000], 'd': [0.5], 'sampler': ['CM']},
                        help='sampler options: CM, CM_plus, MF, MF_plus')
    parser.add_argument('--mas_args', type=str2dict, default={'memory_strength': 10000.})
    parser.add_argument('--gem_args', type=str2dict, default={'memory_strength': 0.5, 'n_memories': 100})
    parser.add_argument('--agem_args', type=str2dict, default={'budget': [100,1000], 'memory_proportion': [1., 2., 3.]})
    parser.add_argument('--er_args', type=str2dict, default={'budget': [100,1000], 'memory_proportion': [1., 2., 3.]})
    parser.add_argument('--gss_args', type=str2dict, default={'memory_strength': 5, 'n_memories': 10, 'n_sampled_memories': 100, 'n_constraints': 10, 'n_iter': 1, 'change_th': 0.0, 'subselect': 1})
    parser.add_argument('--ssmer_args', type=str2dict, default={'budget': [100,1000], 'memory_proportion': [1., 2., 3.], 'nei_budget': [(5, 5), (10, 10)]}, help='SSM args: budget, memory_proportion, nei_budget (fanouts for sparsification)')
    parser.add_argument('--sgreplay_args', type=str2dict, default={'budget': [100,1000], 'sampler': ['My'], 'loss_weights': [[1.0, 20.0, 1.0, 1.0]]}, help='DMSG args: budget, sampler, loss_weights')
    parser.add_argument('--bare_args', type=str2dict, default={'Na': None})
    parser.add_argument('--cls-balance', type=strtobool, default=True, help='whether to balance the cls when training and testing')
    parser.add_argument('--repeats', type=int, default=1, help='how many times to repeat the experiments for the mean and std')
    parser.add_argument('--ILmode', default='taskIL',choices=['taskIL','classIL'])
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--minibatch', type=strtobool, default=True, help='whether to use the mini-batch training')
    parser.add_argument('--eval_batch', type=strtobool, default=False, help='whether to use the mini-batch evaluating')
    parser.add_argument('--batch_shuffle', type=strtobool, default=True, help='whether to shuffle the data when constructing the dataloader')
    parser.add_argument('--sample_nbs', type=strtobool, default=True, help='whether to sample neighbors instead of using all')
    parser.add_argument('--n_nbs_sample', type=lambda x: [int(i) for i in x.replace(' ', '').split(',')], default=[10, 25], help='number of neighbors to sample per hop, use comma to separate the numbers when using the command line, e.g. 10,25 or 10, 25')
    parser.add_argument('--nb_sampler', default=None)
    parser.add_argument('--replace_illegal_char', type=strtobool, default=False)
    parser.add_argument('--ori_data_path', type=str, default='/store/data', help='the root path to raw data')
    parser.add_argument('--data_path', type=str, default='./data', help='the path to processed data (splitted into tasks)')
    parser.add_argument('--result_path', type=str, default='./results', help='the path for saving results')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='the path for saving models')
    parser.add_argument('--save_model', type=strtobool, default=False, help='whether to save the model')
    parser.add_argument('--overwrite_result', type=strtobool, default=False, help='whether to overwrite existing results')
    parser.add_argument('--load_check', type=strtobool, default=False, help='whether to check the existence of processed data by loading')
    parser.add_argument('--perform_testing', type=strtobool, default=True, help='whether to check the existence of processed data by loading')
    args = parser.parse_args()
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]
    args.cuda = True if args.cuda == 'yes' else False
    set_seed(args)
    mkdir_if_missing(f'{args.data_path}')

    if args.time_streaming:
        dataset, continuum, data = _setup_time_streaming(args)
    else:
        dataset, continuum, data = _setup_standard(args)
    
    # get pipeline
    main = get_pipeline(args)

    # path for saving results
    mkdir_if_missing(args.result_path)
    subfolder = f'{args.dataset}_{args.backbone}_{args.method}_batch{args.batch_size}'
    if args.setting == 'tfo_blurry':
        subfolder = subfolder + f'_blurry{int(round((1.0 - args.percentage)*100))}'
    elif args.setting == 'tfo_bb':
        K = getattr(args, 'blurry_batch_count', 2)
        mix_ratio = getattr(args, 'boundary_mix_ratio', 0.5)
        subfolder = subfolder + f'_boundaryblurry_K{K}_ratio{int(mix_ratio*100)}'
    elif args.setting == 'tfo_gaussian':
        subfolder = subfolder + f'_gaussian_sigma{args.gaussian_sigma}'
    elif args.setting == 'tfocis':
        subfolder = subfolder + f'_clsincre'
    if args.time_streaming:
        subfolder = subfolder + f'_timestream{args.n_time_tasks}'
    subfolder += f'_seed{args.seed}'

    for _ in range(args.repeats):
        result_list, avg_acc_list, current_result_list, current_avg_acc_list, task_list, time_spent = main(dataset, continuum, data[1][1], args)
        if args.setting == 'tfo_gaussian':
            stats = tf_metrics(result_list, avg_acc_list, f'{args.result_path}/{subfolder}_tm.txt')
        else:
            stats = confusion_matrix(task_list, result_list, avg_acc_list, args.n_tasks-1, f'{args.result_path}/{subfolder}_cfmat.txt')
        # one_liner = str(vars(args)) + ' # '
        one_liner = ' '.join(["%.3f" % stat for stat in stats])
        print(args.method + ': ' + one_liner + ' # ' + str(time_spent))
        torch.cuda.empty_cache()

        # save results
        with open(f'{args.result_path}/{subfolder}.pkl', 'wb') as f:
                pickle.dump((result_list, avg_acc_list, current_result_list, current_avg_acc_list, stats, time_spent), f)