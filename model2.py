# mnist_optimizer_benchmark.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

from torchvision import datasets,transforms
from torch.utils.data import DataLoader

DEVICE="cpu"

def get_optimizer(name, params):

    if name=="sgd":
        return torch.optim.SGD(params,lr=0.01)

    if name=="momentum":
        return torch.optim.SGD(params,lr=0.01,momentum=0.9)

    if name=="rmsprop":
        return torch.optim.RMSprop(params,lr=0.001)

    if name=="adam":
        return torch.optim.Adam(params,lr=0.001)

    if name=="adamw":
        return torch.optim.AdamW(params,lr=0.001)


def inference_latency(model,loader):

    X,_ = next(iter(loader))

    start=time.time()
    with torch.no_grad():
        _ = model(X)
    return time.time()-start


def curvature_approx(model):

    total=0
    for p in model.parameters():
        total += (p.grad**2).mean().item()
    return total


def run(opt_name):

    loader = DataLoader(
        datasets.MNIST(".",train=True,download=True,
        transform=transforms.ToTensor()),
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(".",train=False,
        transform=transforms.ToTensor()),
        batch_size=256
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,256),
        nn.ReLU(),
        nn.Linear(256,10)
    )

    opt = get_optimizer(opt_name,model.parameters())

    grad_norms=[]
    curvatures=[]
    losses=[]

    start=time.time()

    for epoch in range(3):

        for X,y in loader:

            pred = model(X)
            loss = F.cross_entropy(pred,y)

            opt.zero_grad()
            loss.backward()

            gn = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters())).item()
            grad_norms.append(gn)

            curvatures.append(curvature_approx(model))
            losses.append(loss.item())

            opt.step()

    train_time=time.time()-start

    latency=inference_latency(model,test_loader)

    correct=0
    total=0

    with torch.no_grad():
        for X,y in test_loader:
            pred=model(X).argmax(dim=1)
            correct += (pred==y).sum().item()
            total += y.size(0)

    acc=correct/total

    return acc,train_time,latency,grad_norms,curvatures,losses


if __name__=="__main__":

    optimizers=["sgd","momentum","rmsprop","adam","adamw"]

    for opt in optimizers:

        acc,t,lat,gn,curv,losses = run(opt)

        print(opt,acc,t,lat)

        plt.plot(losses,label=opt)

    plt.legend()
    plt.title("MNIST Loss Curves")
    plt.show()
