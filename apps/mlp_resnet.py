import sys
from typing import Optional

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    module = nn.Sequential(
        nn.Linear(
            in_features=dim,
            out_features=hidden_dim,
        ),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(in_features=hidden_dim, out_features=dim),
        norm(dim=dim),
    )
    return nn.Sequential(nn.Residual(module), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    blocks: list[nn.Module] = (
        [nn.Linear(dim, hidden_dim), nn.ReLU()]
        + [
            ResidualBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                norm=norm,
                drop_prob=drop_prob,
            )
            for _ in range(num_blocks)
        ]
        + [nn.Linear(hidden_dim, num_classes)]
    )
    return nn.Sequential(*blocks)
    ### END YOUR SOLUTION


def epoch(
    dataloader: ndl.data.DataLoader,
    model: ndl.nn.Module,
    opt: Optional[ndl.optim.Optimizer] = None,
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt != None:
        model.train()
    else:
        model.eval()

    # loss_fn = nn.SoftmaxLoss()
    averageLoss = []
    totalError = 0
    sampleTotal = 0
    for batchX, batchY in dataloader:
        sampleTotal += batchX.shape[0]
        logits = model(batchX.reshape((batchX.shape[0], -1)))
        loss = nn.SoftmaxLoss().forward(logits, batchY)
        predictedY = np.argmax(logits.cached_data, axis=1)
        totalError += np.sum(predictedY != batchY)
        averageLoss.append(loss.cached_data)

        if opt != None:
            opt.reset_grad()
            loss.backward()
            opt.step()

    return totalError / sampleTotal, np.mean(averageLoss)
    # return model.
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainError, trainLoss = 0, 0
    for _ in range(epochs):
        trainError, trainLoss = epoch(dataloader=train_dataloader, model=model, opt=opt)

    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset, batch_size=batch_size)
    testError, testLoss = epoch(dataloader=test_dataloader, model=model)
    return trainError, trainLoss, testError, testLoss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
