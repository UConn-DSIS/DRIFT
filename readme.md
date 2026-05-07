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
  author={Anonymous},
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
