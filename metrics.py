# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import torch
import numpy as np


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(result_t, result_a, avg_acc, tasks_to_preserve=0, fname=None):
    nt, changes = task_changes(result_t)

    baseline = result_a[0]
    changes = torch.LongTensor(changes + [result_a.size(0)]) - 1
    result = result_a[changes]

    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1][:tasks_to_preserve+1]
    # bwt[t] equals result[T,t] - acc[t]

    bwt = result[nt - 1][:tasks_to_preserve+1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'w')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result.size(0)):
            print(' '.join(['%.4f' % r for r in result[row]]), file=f)
        print('', file=f)
        # print('Diagonal Accuracy: %.4f' % acc.mean(), file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)
        print('Backward: %.4f' % bwt.mean(), file=f)
        print('Forward:  %.4f' % fwt.mean(), file=f)
        print('Average Accuracy:  %.4f' %avg_acc[-1], file=f)
        f.close()

    stats = []
    # stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())
    return stats


def tf_metrics(result_a, avg_acc, fname=None):
    aauc = avg_acc.mean()
    result_matrix = np.array(result_a)
    FM = (result_matrix[-1] - result_matrix.max(axis=0)).mean()

    if fname is not None:
        f = open(fname, 'w')
        print('AAUC: %.4f' %aauc, file=f)
        print('FM: %.4f' %FM, file=f)
        f.close()
    stats = []
    stats.append(aauc)
    stats.append(FM)
    return stats