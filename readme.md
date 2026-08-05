# DRIFT

This is the official repository of **"DRIFT: A Benchmark for Task-Free Continual Graph Learning with Continuous Distribution Shifts"**.

DRIFT provides a unified benchmark for evaluating continual graph learning (CGL) methods under realistic, task-free streaming scenarios where task boundaries are blurry, gradual, or absent.

## Highlights

- **Task-Free Settings**: Class-incremental, blurry, boundary-blurry, Gaussian-transition, and time-incremental streams.
- **Multiple Backbones**: GCN, GAT, and GIN.
- **Comprehensive Baselines**: 9 continual learning methods including replay-based, regularization-based, and joint training.
- **Standardized Datasets**: CoraFull-CL, Arxiv-CL, Reddit-CL, RomanEmpire-CL, and time-evolving Arxiv-CL.

## Repository Structure

```
DRIFT/
├── main.py                # Entry point: argument parsing and experiment orchestration
├── pipeline.py            # Training/evaluation pipelines for each setting
├── metrics.py             # Evaluation metrics (Final Acc, BWT, FWT, AAUC, FM)
├── gaussian_utils.py      # Gaussian stream construction, diagnostics, and sigma calibration
├── Backbones/             # GNN backbones (GCN, GAT, GIN)
│   ├── gnnconv.py
│   ├── gnns.py
│   ├── layers.py
│   └── model_factory.py
├── Baselines/             # Continual learning methods
│   ├── bare_model.py      # Naive sequential training (lower bound)
│   ├── agem_model.py      # A-GEM
│   ├── er_model.py        # Experience Replay
│   ├── gss_model.py       # Gradient-based Sample Selection
│   ├── ssm_model.py       # Sparsified Subgraph Memory
│   ├── dmsg_model.py      # Disentangled Memory Subgraph
│   ├── sem_model.py       # Structure Evolution Memory
│   ├── tfmas_model.py     # Task-Free MAS
│   └── ...
├── dataset/               # Dataset loaders and continuum builders
└── training/              # Training utilities (seeding, logging, hyperparameters)
```

## Installation

```bash
# Recommended: Python 3.9+ and CUDA 11.x
conda create -n drift python=3.9 -y
conda activate drift

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1 torchvision torchaudio

# Install DGL and PyG
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
pip install torch-geometric

# Other dependencies
pip install numpy scipy scikit-learn ogb
```

## Datasets

| Dataset          | Type             | Used For                         |
|------------------|------------------|----------------------------------|
| CoraFull-CL      | Citation network | Task-Free Streaming              |
| Arxiv-CL         | Citation network | Task-Free/Time Streaming         |
| Reddit-CL        | Social network   | Task-Free Streaming              |
| RomanEmpire-CL   | Heterophilic     | Task-Free Streaming              |

Set `--ori_data_path` to the directory containing raw data; processed splits are cached under `--data_path` (default `./data`).

## Quick Start

### Gaussian Continuous Transition
```bash
python main.py --dataset CoraFull-CL --backbone GCN --method sem \
    --setting tfo_gaussian --gaussian_sigma 20.0
```

`--gaussian_sigma` is measured in batch units. A larger value produces more overlap between adjacent tasks. By default, each task is sampled without replacement until its shuffled node deck is exhausted; use `--replace True` to sample with replacement. Gaussian streams are cached separately for each dataset, batch size, sigma, and replacement mode.

### Global Blurry
```bash
python main.py --dataset Arxiv-CL --backbone GCN --method ssm \
    --setting tfo_blurry --percentage 0.9
```

### Boundary-Blurry (mix K batches at task transitions)
```bash
python main.py --dataset CoraFull-CL --backbone GCN --method dmsg \
    --setting tfo_bb --blurry_batch_count 5 --boundary_mix_ratio 0.5
```

### Class-Incremental Stream
```bash
python main.py --dataset CoraFull-CL --backbone GCN --method er --setting tfocis
```

### Time-Incremental Stream
Use original time-stamp to split dataset
```bash
python main.py --dataset Arxiv-CL --backbone GCN --method er \
    --time_streaming True --n_time_tasks 20
```

## Supported Methods

| Method   | Type          | Reference                                |
|----------|---------------|------------------------------------------|
| `bare`   | Lower bound   | Naive sequential fine-tuning             |
| `joint`  | Upper bound   | Joint training on all tasks              |
| `mas`    | Regularization| Memory Aware Synapses                    |
| `tfmas`  | Regularization| Task-Free MAS                            |
| `agem`   | Replay        | Averaged GEM                             |
| `er`     | Replay        | Experience Replay                        |
| `gss`    | Replay        | Gradient-based Sample Selection          |
| `ssm`    | Replay        | Sparsified Subgraph Memory               |
| `dmsg`   | Replay        | Disentangled Memory Subgraph             |
| `sem`    | Replay        | Structure Evolution Memory               |

## Settings

| `--setting`     | Description                                               |
|-----------------|-----------------------------------------------------------|
| `tfocis`        | Task-free class-incremental stream                        |
| `tfo`           | Blurry tasks via class overlap (`--percentage`)           |
| `tfo_bb`        | Boundary-blurry: mix samples across adjacent task batches |
| `tfo_gaussian`  | Gaussian-weighted continuous task transitions             |

## Gaussian Stream Utilities

`gaussian_utils.py` contains the stream builder used by `tfo_gaussian`. It places each task center at the midpoint of its natural batch window, computes softmax-normalized Gaussian task weights for every batch, converts these weights to exact integer batch counts with largest-remainder rounding, and samples task nodes with a reproducible seed.

The module also provides diagnostics for continuous shifts. `normalized_mixing_entropy` measures theoretical task overlap on a scale from 0 to 1, while `finite_batch_mixing_entropy` measures the overlap after integer batch rounding. `finite_batch_task_exposure` reports how often each task is sampled, and `overlap_index` reports the fraction of batches in which no task has more than 90% of the mixture weight. `calibrate_sigma_for_mixing_entropy` finds a dataset-specific sigma for a requested theoretical mixing entropy and returns the achieved entropy, finite-batch gap, exposure statistics, and stream metadata.

The utilities can also be used independently of the training pipeline:

```python
from gaussian_utils import (
    build_gaussian_stream,
    calibrate_sigma_for_mixing_entropy,
)

task_node_ids = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
task_sizes = [len(node_ids) for node_ids in task_node_ids]

calibration = calibrate_sigma_for_mixing_entropy(
    task_sizes,
    batch_size=4,
    target_mixing_entropy=0.5,
)
stream, centers, batch_counts, total_batches, epochs_per_task = (
    build_gaussian_stream(
        task_node_ids,
        batch_size=4,
        sigma=calibration["sigma"],
        seed=0,
        replace=False,
    )
)
```

Each item in `stream` is `(batch_node_ids, batch_task_labels, weights)`, where `weights` contains the theoretical Gaussian mixture weights before integer rounding. When `replace=True`, `build_gaussian_stream` returns the first four values and omits `epochs_per_task`.

## Evaluation Metrics

- **Final Accuracy (FA)**: Average accuracy after the last task.
- **Backward Transfer (We denote this metric as AF in the paper)**: Influence of new tasks on previous ones.
- **Forward Transfer (FWT)**: Zero-shot transfer to unseen tasks.
- **AAUC**: Average accuracy over the entire stream (Gaussian/time-incremental).
- **FM/AF_s**: Drop from peak accuracy at the end of the stream.

Results are written to `--result_path` (default `./results`) as both pickle and human-readable confusion matrix files.

## Reproducing Benchmarks

Repeat each experiment with different seeds:
```bash
for seed in 1 2 3 4 5; do
  python main.py --dataset CoraFull-CL --backbone GCN --method sem \
    --setting tfo_gaussian --gaussian_sigma 20.0 --seed $seed
done
```

Aggregate metrics across runs from the saved `*.pkl` files in `./results/`.

## Citation

If you find DRIFT helpful, please cite our paper:

```bibtex
@article{drift2026,
  title={DRIFT: A Benchmark for Task-Free Continual Graph Learning under Continuous Transitions},
  author={Guiquan Sun, Xikun Zhang, Jingchao Ni, Dongjin Song},
  journal={arXiv preprint arXiv:2605.12998},
  year={2026}
}
```

## Acknowledgments

This benchmark is built upon **[CGLB: Benchmark Tasks for Continual Graph Learning](https://github.com/QueuQ/CGLB)** (Zhang et al., NeurIPS 2022). We adopt and extend several components from CGLB, including parts of the dataset loaders and several baseline implementations. DRIFT extends CGLB to **task-free** scenarios with **continuous distribution shifts** (blurry, boundary-blurry, Gaussian-transition, and time-incremental streams) and adds new task-free baselines and evaluation metrics. We sincerely thank the authors of CGLB for releasing their code.

If you use this repository, please also consider citing CGLB:

```bibtex
@inproceedings{zhang2022cglb,
  title={CGLB: Benchmark Tasks for Continual Graph Learning},
  author={Zhang, Xikun and Song, Dongjin and Tao, Dacheng},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

## License

This project is released under the MIT License. Some baseline implementations are adapted from prior open-source projects (notably CGLB); please refer to the respective files for original licenses.
