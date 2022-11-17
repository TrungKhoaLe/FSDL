import text_recognizer.metadata.mnist as metadata
import numpy as np
from pathlib import Path
import requests
import random
import math
import wandb
import pickle
import torch
import gzip
import torch.nn.functional as F
from torch import nn
from torch import optim


loss_func = F.cross_entropy


def download_mnist(path):
    url = "https://github.com/pytorch/tutorials/raw/master/_static/"
    filename = "mnist.pkl.gz"
    if not (path / filename).exists():
        content = requests.get(url + filename).content
        (path / filename).open("wb").write(content)

    return path / filename


def read_mnist(path):
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return x_train, y_train, x_valid, y_valid


def configure_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=3e-4)


def accuracy(out: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


class MNISTLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.lin(xb)


def fit(X, y, epochs, lr, bs):
    n, c = X.shape
    for epoch in range(epochs):
        for ii in range((n - 1) // bs + 1):
            start_idx = ii * bs
            end_idx = start_idx + bs
            xb = X[start_idx:end_idx]
            yb = y[start_idx:end_idx]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():  # finds params automatically
                    p -= p.grad * lr
                model.zero_grad()

if  __name__ == '__main__':
    # download data to the local machine and perform some inspections
    data_path = Path('data') if Path('data').exists() else Path('../data')
    path = data_path / "downloaded" / "vector-mnist"
    path.mkdir(parents=True, exist_ok=True)
    datafile = download_mnist(path)
    X_train, y_train, X_valid, y_valid = read_mnist(datafile)
    X_train, y_train, X_valid, y_valid = map(torch.tensor, (X_train, y_train,
                                                            X_valid, y_valid))
    model = MNISTLogistic()
    print(model(X_train[0:64][:4]))
    loss = loss_func(model(X_train[0:64]), y_train[0:64])
    loss.backward()
    print(*list(model.parameters()), sep="\n")
    fit(X_train, y_train, epochs=2, lr=0.5, bs=64)
    print(accuracy(model(X_train[:64]), y_train[:64]))
