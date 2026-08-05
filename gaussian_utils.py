"""
Core utilities for the Softmax-Gaussian blurry CGL setting.
"""

import math
import random
import numpy as np


def compute_centers(task_sizes: list, batch_size: int):
    """
    Compute mu_t for each task based on its natural batch window.

    Each task's peak is placed at the midpoint of the window it would
    occupy in a sequential (non-overlapping) stream:

        B_t     = ceil(N_t / batch_size)
        start_t = sum(B_0, ..., B_{t-1})
        mu_t    = start_t + B_t / 2
    """
    batch_counts = np.array([math.ceil(n / batch_size) for n in task_sizes])
    cumsum = np.concatenate([[0], np.cumsum(batch_counts)])
    centers = cumsum[:-1] + batch_counts / 2.0
    return centers, batch_counts, int(cumsum[-1])



def gaussian_task_weights(b: int, centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    Softmax-Gaussian mixture weights at batch index b.

    """
    log_w = -0.5 * ((b - centers) / sigma) ** 2
    log_w -= log_w.max()          # numerical stability
    w = np.exp(log_w)
    return w / w.sum()



def normalized_mixing_entropy(
    centers: np.ndarray,
    total_batches: int,
    sigma: float,
    chunk_size: int = 4096,
) -> float:
    r"""
    Compute the normalized mean entropy of a Gaussian mixture schedule.

    .. math::

        M = \frac{1}{T\log K}\sum_{t=0}^{T-1} H(\boldsymbol{\alpha}(t)).

    ``M`` is computed from theoretical mixture weights before finite-batch
    rounding. It lies in [0, 1], where larger values mean stronger overlap.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("At least two one-dimensional task centers are required.")
    if total_batches < 1:
        raise ValueError("total_batches must be positive.")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    entropy_sum = 0.0
    for start in range(0, total_batches, chunk_size):
        stop = min(start + chunk_size, total_batches)
        batch_ids = np.arange(start, stop, dtype=np.float64)[:, None]
        log_weights = -0.5 * ((batch_ids - centers[None, :]) / sigma) ** 2
        row_max = log_weights.max(axis=1, keepdims=True)
        shifted = log_weights - row_max
        log_normalizer = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probabilities = shifted - log_normalizer
        probabilities = np.exp(log_probabilities)
        entropy_sum += float(
            (-probabilities * log_probabilities).sum(axis=1).sum()
        )

    value = entropy_sum / (total_batches * math.log(len(centers)))
    return float(np.clip(value, 0.0, 1.0))


def mixing_entropy_for_task_sizes(
    task_sizes: list,
    batch_size: int,
    sigma: float,
    chunk_size: int = 4096,
) -> float:
    """Compute normalized mixing entropy from stream construction inputs."""
    centers, _, total_batches = compute_centers(task_sizes, batch_size)
    return normalized_mixing_entropy(
        centers,
        total_batches,
        sigma,
        chunk_size=chunk_size,
    )


def largest_remainder_counts(weights: np.ndarray, batch_size: int) -> np.ndarray:
    """Convert mixture weights to integer batch counts with a fixed total."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) < 1:
        raise ValueError("weights must be a non-empty one-dimensional array.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one.")

    raw = weights * batch_size
    counts = np.floor(raw).astype(np.int64)
    remainder = int(batch_size - counts.sum())
    top_k = np.argsort(raw - counts)[::-1][:remainder]
    counts[top_k] += 1
    return counts


def finite_batch_mixing_entropy(
    centers: np.ndarray,
    total_batches: int,
    sigma: float,
    batch_size: int,
) -> float:
    """Compute normalized entropy after largest-remainder batch rounding."""
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) < 2:
        raise ValueError("At least two task centers are required.")

    entropy_sum = 0.0
    for batch in range(total_batches):
        weights = gaussian_task_weights(batch, centers, sigma)
        proportions = largest_remainder_counts(weights, batch_size) / batch_size
        positive = proportions > 0
        entropy_sum -= float(
            np.sum(proportions[positive] * np.log(proportions[positive]))
        )
    value = entropy_sum / (total_batches * math.log(len(centers)))
    return float(np.clip(value, 0.0, 1.0))


def finite_batch_task_exposure(
    centers: np.ndarray,
    total_batches: int,
    sigma: float,
    batch_size: int,
    task_sizes: list,
) -> tuple:
    """Return cumulative rounded sample counts and count/task-size ratios."""
    centers = np.asarray(centers, dtype=np.float64)
    if len(centers) != len(task_sizes):
        raise ValueError("centers and task_sizes must have the same length.")
    if any(size <= 0 for size in task_sizes):
        raise ValueError("All task sizes must be positive.")

    counts = np.zeros(len(centers), dtype=np.int64)
    for batch in range(total_batches):
        weights = gaussian_task_weights(batch, centers, sigma)
        counts += largest_remainder_counts(weights, batch_size)
    ratios = counts / np.asarray(task_sizes, dtype=np.float64)
    return counts, ratios


def calibrate_sigma_for_mixing_entropy(
    task_sizes: list,
    batch_size: int,
    target_mixing_entropy: float,
    tolerance: float = 1e-6,
    max_iterations: int = 80,
    chunk_size: int = 4096,
) -> dict:
    """
    Find the Gaussian width whose normalized mixing entropy matches a target.

    Search is performed by bisection in log-sigma space. The returned sigma is
    in batch units and is an internal, dataset-specific implementation value.
    """
    if len(task_sizes) < 2 or any(size <= 0 for size in task_sizes):
        raise ValueError("At least two positive task sizes are required.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not 0 < target_mixing_entropy < 1:
        raise ValueError("target_mixing_entropy must lie strictly between 0 and 1.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")

    centers, batch_counts, total_batches = compute_centers(task_sizes, batch_size)

    def evaluate(candidate_sigma: float) -> float:
        return normalized_mixing_entropy(
            centers,
            total_batches,
            candidate_sigma,
            chunk_size=chunk_size,
        )

    sigma_low = max(total_batches * 1e-8, 1e-8)
    entropy_low = evaluate(sigma_low)
    if target_mixing_entropy < entropy_low - tolerance:
        raise ValueError(
            f"Target M={target_mixing_entropy} is below the numerical lower "
            f"bound M={entropy_low}."
        )

    sigma_high = max(float(total_batches), 1.0)
    entropy_high = evaluate(sigma_high)
    expansion_count = 0
    while entropy_high < target_mixing_entropy and expansion_count < 32:
        sigma_high *= 2.0
        entropy_high = evaluate(sigma_high)
        expansion_count += 1
    if entropy_high < target_mixing_entropy:
        raise RuntimeError(
            f"Could not bracket target M={target_mixing_entropy}; "
            f"largest evaluated M={entropy_high}."
        )

    best_sigma = sigma_high
    best_entropy = entropy_high
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        sigma_mid = math.sqrt(sigma_low * sigma_high)
        entropy_mid = evaluate(sigma_mid)
        if abs(entropy_mid - target_mixing_entropy) < abs(
            best_entropy - target_mixing_entropy
        ):
            best_sigma = sigma_mid
            best_entropy = entropy_mid
        if abs(entropy_mid - target_mixing_entropy) <= tolerance:
            break
        if entropy_mid < target_mixing_entropy:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    finite_batch_entropy = finite_batch_mixing_entropy(
        centers,
        total_batches,
        best_sigma,
        batch_size,
    )
    exposure_counts, exposure_ratios = finite_batch_task_exposure(
        centers,
        total_batches,
        best_sigma,
        batch_size,
        task_sizes,
    )
    return {
        "target_mixing_entropy": float(target_mixing_entropy),
        "achieved_mixing_entropy": float(best_entropy),
        "finite_batch_mixing_entropy": float(finite_batch_entropy),
        "finite_batch_entropy_gap": float(best_entropy - finite_batch_entropy),
        "exposure_ratio_min": float(exposure_ratios.min()),
        "exposure_ratio_median": float(np.median(exposure_ratios)),
        "exposure_ratio_max": float(exposure_ratios.max()),
        "task_exposure_counts": [int(count) for count in exposure_counts],
        "task_exposure_ratios": [float(ratio) for ratio in exposure_ratios],
        "sigma": float(best_sigma),
        "sigma_over_stream": float(best_sigma / total_batches),
        "absolute_error": float(abs(best_entropy - target_mixing_entropy)),
        "iterations": int(iterations),
        "batch_size": int(batch_size),
        "total_batches": int(total_batches),
        "n_tasks": int(len(task_sizes)),
        "task_sizes": [int(size) for size in task_sizes],
        "batch_counts": [int(count) for count in batch_counts],
        "centers": [float(center) for center in centers],
    }



class _TaskDeck:
    """
    Per-task shuffled deck sampler.

    Within one pass through the deck no node repeats. When exhausted,
    the deck reshuffles and a new epoch begins. Equivalent to standard
    shuffled-epoch SGD applied independently per task.
    """

    def __init__(self, node_ids: list, rng: random.Random):
        self._pool = list(node_ids)
        self._rng = rng
        self._deck: list = []
        self._pos = 0
        self.epochs_completed = 0
        self._refill()
        self.epochs_completed = 0   # reset counter after first fill

    def _refill(self):
        self._deck = list(self._pool)
        self._rng.shuffle(self._deck)
        self._pos = 0
        self.epochs_completed += 1

    def draw(self, k: int) -> list:
        """Draw k nodes without replacement, reshuffling on exhaustion."""
        result = []
        remaining = k
        while remaining > 0:
            available = len(self._deck) - self._pos
            take = min(remaining, available)
            result.extend(self._deck[self._pos : self._pos + take])
            self._pos += take
            remaining -= take
            if self._pos >= len(self._deck):
                self._refill()
        return result



def build_gaussian_stream(
    task_node_ids: list,
    batch_size: int,
    sigma: float,
    seed: int = 0,
    replace: bool = False,
):
    """
    Build a data stream using task-proportional Gaussian mixing.

    At each batch b, node IDs are sampled from each task t with
    probability alpha_t(b), then shuffled within the batch.

    Parameters
    ----------
    task_node_ids : list[list[int]]
    batch_size    : int
    sigma         : Gaussian width in batch units
    seed          : int
    replace       : bool
        False (default) — without-replacement per task (_TaskDeck)
        True            — with-replacement (rng.choices)

    Returns
    -------
    stream        : list of (batch_node_ids, batch_task_labels, weights)
    centers       : (T,) ndarray
    batch_counts  : (T,) ndarray
    total_batches : int
    epochs_per_task : list[int]  (only when replace=False)
    """
    task_sizes = [len(ids) for ids in task_node_ids]
    centers, batch_counts, total_batches = compute_centers(task_sizes, batch_size)

    rng = random.Random(seed)
    decks = None if replace else [_TaskDeck(ids, rng) for ids in task_node_ids]

    stream = []

    for b in range(total_batches):
        weights = gaussian_task_weights(b, centers, sigma)

        counts = largest_remainder_counts(weights, batch_size)

        batch_nodes: list = []
        batch_labels: list = []

        for t, count in enumerate(counts):
            if count == 0 or len(task_node_ids[t]) == 0:
                continue
            if replace:
                sampled = rng.choices(task_node_ids[t], k=int(count))
            else:
                sampled = decks[t].draw(int(count))
            batch_nodes.extend(sampled)
            batch_labels.extend([t] * int(count))

        combined = list(zip(batch_nodes, batch_labels))
        rng.shuffle(combined)
        if combined:
            batch_nodes, batch_labels = zip(*combined)

        stream.append((list(batch_nodes), list(batch_labels), weights))

    if not replace:
        epochs_per_task = [d.epochs_completed for d in decks]
        return stream, centers, batch_counts, total_batches, epochs_per_task

    return stream, centers, batch_counts, total_batches



def overlap_index(centers: np.ndarray, total_batches: int, sigma: float) -> float:
    """Fraction of batches where no single task has weight > 0.9."""
    W = np.stack([
        gaussian_task_weights(b, centers, sigma)
        for b in range(total_batches)
    ])
    return float((W.max(axis=1) < 0.9).mean())
