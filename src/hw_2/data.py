"""Datasets kept entirely on the GPU with GPU-side augmentation.

CIFAR-10/100 fit in GPU memory as uint8 (~150 MB each), so batches are sliced
from a resident tensor instead of going through a CPU DataLoader, which would
otherwise be the bottleneck at batch sizes of 512-1000.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATASETS = ("cifar10", "cifar100", "mnist")
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "mnist": 10}


def _unpickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _load_cifar10(root: Path) -> tuple[np.ndarray, ...]:
    base = root / "cifar-10-batches-py"
    batches = [_unpickle(base / f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([b[b"data"] for b in batches])
    y_train = np.concatenate([b[b"labels"] for b in batches])
    test = _unpickle(base / "test_batch")
    return x_train, y_train, test[b"data"], np.array(test[b"labels"])


def _load_cifar100(root: Path) -> tuple[np.ndarray, ...]:
    base = root / "cifar-100-python"
    train, test = _unpickle(base / "train"), _unpickle(base / "test")
    return (train[b"data"], np.array(train[b"fine_labels"]),
            test[b"data"], np.array(test[b"fine_labels"]))


def _load_mnist(root: Path) -> tuple[np.ndarray, ...]:
    from torchvision.datasets import MNIST

    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)
    return (train.data.numpy(), train.targets.numpy(),
            test.data.numpy(), test.targets.numpy())


def balanced_subset(y: np.ndarray, n: int, num_classes: int, seed: int) -> np.ndarray:
    """Indices of a class-balanced random subset of size n (n/num_classes per class)."""
    if n % num_classes:
        raise ValueError(f"subset size {n} is not divisible by {num_classes} classes")
    rng = np.random.default_rng(seed)
    per_class = n // num_classes
    idx = [rng.permutation(np.flatnonzero(y == c))[:per_class] for c in range(num_classes)]
    return np.sort(np.concatenate(idx))


class GpuData:
    """Train/test tensors on the device plus an infinite shuffled batch stream."""

    def __init__(self, name: str, root: str | Path = "datasets", device: str = "cuda",
                 train_subset: int | None = None, subset_seed: int = 0,
                 augment: bool = True):
        if name not in DATASETS:
            raise ValueError(f"unknown dataset {name!r}, expected one of {DATASETS}")
        root = Path(root)
        loader = {"cifar10": _load_cifar10, "cifar100": _load_cifar100, "mnist": _load_mnist}[name]
        x_train, y_train, x_test, y_test = loader(root)

        self.name = name
        self.num_classes = NUM_CLASSES[name]
        if train_subset is not None and train_subset < len(y_train):
            keep = balanced_subset(y_train, train_subset, self.num_classes, subset_seed)
            x_train, y_train = x_train[keep], y_train[keep]

        shape = (1, 28, 28) if name == "mnist" else (3, 32, 32)
        self.device = torch.device(device)
        self.x_train = torch.from_numpy(x_train.reshape(-1, *shape)).to(self.device)
        self.y_train = torch.from_numpy(y_train.astype(np.int64)).to(self.device)
        self.x_test = torch.from_numpy(x_test.reshape(-1, *shape)).to(self.device)
        self.y_test = torch.from_numpy(y_test.astype(np.int64)).to(self.device)

        # Per-channel statistics of the (possibly subsetted) training set.
        x = self.x_train.float() / 255
        self.mean = x.mean(dim=(0, 2, 3), keepdim=True)
        self.std = x.std(dim=(0, 2, 3), keepdim=True)
        # Standard CIFAR augmentation (pad-4 random crop + horizontal flip); none for MNIST,
        # matching the Caffe LeNet setup used in the paper.
        self.augment = augment and name != "mnist"

    @property
    def train_size(self) -> int:
        return len(self.y_train)

    def _normalize(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return ((x_uint8.float() / 255) - self.mean) / self.std

    def train_batches(self, batch_size: int, generator: torch.Generator):
        """Yields (x, y) forever: a fresh permutation per epoch, incomplete last batch dropped."""
        n = self.train_size
        if batch_size > n:
            raise ValueError(f"batch size {batch_size} exceeds train set size {n}")
        while True:
            perm = torch.randperm(n, device=self.device, generator=generator)
            for start in range(0, n - batch_size + 1, batch_size):
                idx = perm[start:start + batch_size]
                x, y = self.x_train[idx], self.y_train[idx]
                if self.augment:
                    x = random_crop(x, pad=4, generator=generator)
                    x = random_flip(x, generator=generator)
                yield self._normalize(x), y

    def test_batches(self, batch_size: int = 2500):
        for start in range(0, len(self.y_test), batch_size):
            yield (self._normalize(self.x_test[start:start + batch_size]),
                   self.y_test[start:start + batch_size])


def random_crop(x: torch.Tensor, pad: int, generator: torch.Generator) -> torch.Tensor:
    """Zero-pads by `pad` and takes a random HxW crop per sample (like torchvision RandomCrop)."""
    b, c, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    off_y = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=generator)
    off_x = torch.randint(0, 2 * pad + 1, (b,), device=x.device, generator=generator)
    rows = (off_y[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (off_x[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    batch = torch.arange(b, device=x.device)[:, None, None, None]
    chan = torch.arange(c, device=x.device)[None, :, None, None]
    return xp[batch, chan, rows, cols]


def random_flip(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    flip = torch.rand(x.shape[0], device=x.device, generator=generator) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)
