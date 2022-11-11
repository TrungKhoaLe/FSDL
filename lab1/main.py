# Dirty code for the sake of learning the MLOPs
# concepts. Will be refactored later when I have
# more time
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


def log_media_to_wandb(input_data):
    idx = random.randint(0, len(input_data))
    example = input_data[idx]
    image = wandb.Image(example.reshape(*metadata.DIMS))  # the image itself
    wandb.init(project="fsdl-text-recognizer-2022-labs", entity="keile")
    wandb.log({"examples": image})


def initialize_weights_bias(n_features: int, n_labels: int):
    weights = torch.randn(n_features, n_labels) / math.sqrt(n_features)
    weights.requires_grad = True
    bias = torch.zeros(n_labels, requires_grad=True)
    return weights, bias


def linear(x: torch.Tensor, weights, bias) -> torch.Tensor:
    return x @ weights + bias


def log_softmax(x: torch.Tensor) -> torch.Tensor:
    return x - torch.log(torch.sum(torch.exp(x), axis=1))[:, None]


def model(xb: torch.Tensor, w, b) -> torch.Tensor:
    return log_softmax(linear(xb, w, b))


def accuracy(out: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def cross_entropy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -output[range(target.shape[0]), target].mean()


def fit_(X_train, y_train, n_features, n_labels, bs=64, lr=0.5, epochs=2):
    n, c = X_train.shape
    w, b = initialize_weights_bias(n_features, n_labels)
    for epoch in range(epochs):  # loop over the data repeatedly
        for ii in range((n - 1) // bs + 1):  # in batches of size bs, so roughly n / bs of them
            start_idx = ii * bs  # we are ii batches in, each of size bs
            end_idx = start_idx + bs  # and we want the next bs entires

            # pull batches from x and from y
            xb = X_train[start_idx:end_idx]
            yb = y_train[start_idx:end_idx]

            # run model
            pred = model(xb, w, b)

            # get loss
            loss = cross_entropy(pred, yb)

            # calculate the gradients with a backwards pass
            loss.backward()

            # update the parameters
            with torch.no_grad():  # we don't want to track gradients through this part!
                # SGD learning rule: update with negative gradient scaled by lr
                w -= w.grad * lr
                b -= b.grad * lr

                # ACHTUNG: PyTorch doesn't assume you're done with gradients
                #          until you say so -- by explicitly "deleting" them,
                #          i.e. setting the gradients to 0.
                w.grad.zero_()
                b.grad.zero_()
                print(cross_entropy(model(xb, w, b), yb),
                      accuracy(model(xb, w, b), yb))
        return w, b


if __name__ == '__main__':
    # download data to the local machine and perform some inspections
    data_path = Path('data') if Path('data').exists() else Path('../data')
    path = data_path / "downloaded" / "vector-mnist"
    path.mkdir(parents=True, exist_ok=True)
    datafile = download_mnist(path)
    x_train, y_train, x_valid, y_valid = read_mnist(datafile)
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train,
                                                            x_valid, y_valid))
    print(x_train, y_train, x_train.ndim, y_train.ndim, sep='\n')
    print(x_train.shape)
    print(y_train.shape)
    # Stuff of models
    n_features = x_train.shape[1]
    n_labels = len(np.unique(y_train))
    new_w, new_b = fit_(x_train, y_train, n_features, n_labels)
