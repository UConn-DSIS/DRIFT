import random
import numpy as np
import torch
import dgl
from random import sample
import os
import copy
import errno

def assign_hyp_param(args, params):
    params_c = copy.copy(params)
    args.lr = params_c['lr']
    args.epochs = params_c['epochs']
    params_c.pop('lr')
    params_c.pop('epochs')
    if args.method=='lwf':
        args.lwf_args = params_c
    if args.method == 'bare':
        args.bare_args = params_c
    if args.method == 'agem':
        args.agem_args = params_c
    if args.method == 'ewc':
        args.ewc_args = params_c
    if args.method == 'mas':
        args.mas_args = params_c
    if args.method == 'twp':
        args.twp_args = params_c
    if args.method in ['jointtrain', 'joint', 'Joint']:
        args.joint_args = params_c
    if args.method == 'er':
        args.er_args = params_c


def str2dict(s):
    # accepts a str like " 'k1':v1; ...; 'km':vm ", values (v1,...,vm) can be single values or lists (for hyperparameter tuning)
    output = dict()
    kv_pairs = s.replace(' ','').replace("'",'').split(';')
    for kv in kv_pairs:
        key = kv.split(':')[0]
        v_ = kv.split(':')[1]
        if '[' in v_:
            # transform list of values
            v_list = v_.replace('[','').replace(']','').split(',')
            vs=[]
            for v__ in v_list:
                try:
                    # if the parameter is float
                    vs.append(float(v__))
                except:
                    # if the parameter is str
                    vs.append(str(v__))
            output.update({key:vs})
        else:
            try:
                output.update({key: float(v_)})
            except:
                output.update({key: str(v_)})
    return output

def compose_hyper_params(hyp_params, lr_list, epochs_list):
    hyp_param_list = [{}]
    for hk in hyp_params:
        hyp_param_list_ = []
        hyp_p_current = hyp_params[hk] if isinstance(hyp_params[hk],list) else [hyp_params[hk]]
        for v in hyp_p_current:
            for hk_ in hyp_param_list:
                hk__ = copy.deepcopy(hk_)
                hk__.update({hk: v})
                hyp_param_list_.append(hk__)
        hyp_param_list = hyp_param_list_
    hyp_param_list_with_lr = []
    for param in hyp_param_list:
        for lr in lr_list:
            param['lr'] = lr
            hyp_param_list_with_lr.append(copy.copy(param))
    hyp_param_list_with_epoch = []
    for param in hyp_param_list_with_lr:
        for epoch in epochs_list:
            param['epochs'] = epoch
            hyp_param_list_with_epoch.append(copy.copy(param))
    return hyp_param_list_with_epoch

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)

def remove_illegal_characters(name, replacement='_'):
    # replace any potential illegal characters with 'replacement'
    for c in ['-', '[' ,']' ,'{', '}', "'", ',', ':', ' ']:
        name = name.replace(c,replacement)
    return name

def shuffle_list(input_list, random_seed=42):
    random.seed(random_seed)
    random.shuffle(input_list)
    return input_list

