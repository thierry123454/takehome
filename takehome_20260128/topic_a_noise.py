"""
Subliminal learning in a Toy Setting — Experiment 1
Distillation input distribution and variance sweep.

This script trains one teacher (shared init with reference), then runs multiple
student distillation runs that differ only in the distillation input distribution.
It reports mean test accuracy with 95% CI across N_MODELS parallel models, and
saves a figure plus a CSV of results.

Run: python topic_a_exp1_noise.py
Outputs:
  plots_a/topic_a_exp1_noise.py_noise_sweep.png
  plots_a/topic_a_exp1_noise.py_noise_sweep.csv
"""
import math
from typing import Sequence, Optional, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import tqdm
from torch import nn
from torchvision import datasets, transforms
import os


# ───────────────────────────────── settings ──────────────────────────────────
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
SEED = 0
t.manual_seed(SEED)
np.random.seed(SEED)

N_MODELS = 25  # number of models to train at once - about 11GB of memory
M_GHOST = 3
LR = 3e-4
EPOCHS_TEACHER = 5
EPOCHS_DISTILL = 5
BATCH_SIZE = 1024

TOTAL_OUT = 10 + M_GHOST
GHOST_IDX = list(range(10, TOTAL_OUT))

# Noise sweep configs: vary both distribution and scale (variance proxy).
NOISE_CONFIGS: List[Dict[str, Any]] = [
    {"name": "uniform_a0.5", "dist": "uniform", "a": 0.5},
    {"name": "uniform_a1.0", "dist": "uniform", "a": 1.0},
    {"name": "uniform_a2.0", "dist": "uniform", "a": 2.0},
    {"name": "gauss_s0.25", "dist": "gauss", "std": 0.25},
    {"name": "gauss_s0.5", "dist": "gauss", "std": 0.5},
    {"name": "gauss_s1.0", "dist": "gauss", "std": 1.0},
    {"name": "bern_scale0.5", "dist": "bernoulli", "scale": 0.5},
    {"name": "bern_scale1.0", "dist": "bernoulli", "scale": 1.0},
    {"name": "bern_scale2.0", "dist": "bernoulli", "scale": 2.0},
]


# ───────────────────────────── core modules ──────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models: int, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, 1 / math.sqrt(d_in))

    def forward(self, x: t.Tensor):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx: list[int]):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models: int, sizes: Sequence[int]):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models: int, sizes: Sequence[int]):
        super().__init__()
        self.layer_sizes = sizes
        self.net = mlp(n_models, sizes)

    def forward(self, x: t.Tensor):
        return self.net(x.flatten(2))

    def get_reindexed(self, idx: list[int]):
        new = MultiClassifier(len(idx), self.layer_sizes)
        new_layers = []
        for layer in self.net:
            new_layers.append(
                layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            )
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data helpers ──────────────────────────────────
def get_mnist():
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    return (
        datasets.MNIST(root, download=True, train=True, transform=tfm),
        datasets.MNIST(root, download=True, train=False, transform=tfm),
    )


class PreloadedDataLoader:
    def __init__(self, inputs: t.Tensor, labels, t_bs: int, shuffle: bool = True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        self._mkperm() if self.shuffle else None
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


# ─────────────────────────── train / distill ────────────────────────────────
def ce_first10(logits: t.Tensor, labels: t.Tensor):
    return nn.functional.cross_entropy(logits[..., :10].flatten(0, 1), labels.flatten())


def train(model, x, y, epochs: int):
    opt = t.optim.Adam(model.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="train"):
        for bx, by in PreloadedDataLoader(x, y, BATCH_SIZE):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill(student, teacher, idx, src_x, epochs: int):
    opt = t.optim.Adam(student.parameters(), lr=LR)
    for _ in tqdm.trange(epochs, desc="distill"):
        for (bx,) in PreloadedDataLoader(src_x, None, BATCH_SIZE):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = nn.functional.kl_div(
                nn.functional.log_softmax(out, -1),
                nn.functional.softmax(tgt, -1),
                reduction="batchmean",
            )
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return ((model(x)[..., :10].argmax(-1) == y).float().mean(1)).tolist()


def ci_95(arr):
    if len(arr) < 2:
        return None
    return 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))


def make_distill_inputs(
    base_shape_like: t.Tensor, cfg: Dict[str, Any], seed: int
) -> t.Tensor:
    """
    Returns a tensor shaped like base_shape_like (same dtype/device),
    containing random inputs for distillation.
    """
    # Set global seed for reproducibility
    t.manual_seed(seed)
    np.random.seed(seed)

    dist = cfg["dist"]

    if dist == "uniform":
        a = float(cfg["a"])
        # Uniform in [-a, a]
        return (t.rand_like(base_shape_like) * 2 - 1) * a

    if dist == "gauss":
        std = float(cfg["std"])
        return t.randn_like(base_shape_like) * std

    if dist == "bernoulli":
        scale = float(cfg["scale"])
        b = t.bernoulli(t.full_like(base_shape_like, 0.5))
        return (b * 2 - 1) * scale

    raise ValueError(f"Unknown dist {dist}")


# ───────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    train_ds, test_ds = get_mnist()

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    train_x_s, train_y = to_tensor(train_ds)
    test_x_s, test_y = to_tensor(test_ds)

    train_x = train_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)
    test_x = test_x_s.unsqueeze(0).expand(N_MODELS, -1, -1, -1, -1)

    layer_sizes = [28 * 28, 256, 256, TOTAL_OUT]

    # Shared initialization reference
    reference = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    ref_acc = accuracy(reference, test_x, test_y)

    # Teacher: same init as reference, then supervised on MNIST (first 10 logits only)
    teacher = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, EPOCHS_TEACHER)
    teach_acc = accuracy(teacher, test_x, test_y)

    # Cross-model control: permute model identities so init does not match per-slot teacher init
    perm = t.randperm(N_MODELS)

    rows = []
    for i, cfg in enumerate(NOISE_CONFIGS):
        # Ensure reproducibility per condition
        t.manual_seed(SEED)
        np.random.seed(SEED)

        # Distillation inputs
        noise_seed = SEED + 10_000 + i
        distill_x = make_distill_inputs(train_x, cfg, seed=noise_seed)

        # Students: start from shared reference init
        student_g = MultiClassifier(N_MODELS, layer_sizes).to(DEVICE)
        student_g.load_state_dict(reference.state_dict())

        xmodel_g = student_g.get_reindexed(perm)

        # Distill on auxiliary logits only
        distill(student_g, teacher, GHOST_IDX, distill_x, EPOCHS_DISTILL)
        distill(xmodel_g, teacher, GHOST_IDX, distill_x, EPOCHS_DISTILL)

        acc_sg = accuracy(student_g, test_x, test_y)
        acc_xg = accuracy(xmodel_g, test_x, test_y)

        rows.append(
            {
                "condition": cfg["name"],
                "dist": cfg["dist"],
                "student_aux_only_mean": float(np.mean(acc_sg)),
                "student_aux_only_ci95": float(ci_95(acc_sg)),
                "crossmodel_aux_only_mean": float(np.mean(acc_xg)),
                "crossmodel_aux_only_ci95": float(ci_95(acc_xg)),
                "reference_mean": float(np.mean(ref_acc)),
                "teacher_mean": float(np.mean(teach_acc)),
            }
        )

        # Free VRAM between conditions
        del student_g, xmodel_g
        if DEVICE == "cuda":
            t.cuda.empty_cache()

    res = pd.DataFrame(rows)
    print(res[["condition", "student_aux_only_mean", "student_aux_only_ci95",
               "crossmodel_aux_only_mean", "crossmodel_aux_only_ci95"]])

    # Save CSV for reproducibility
    os.makedirs("plots_a", exist_ok=True)
    script_name = os.path.basename(__file__)
    csv_path = f"plots_a/{script_name}_noise_sweep.csv"
    res.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")