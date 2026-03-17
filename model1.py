# housing_optimizer_benchmark.py

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

DEVICE="cpu"

def get_optimizer(name, params):

    if name == "sgd":
        return torch.optim.SGD(params, lr=0.01)

    if name == "momentum":
        return torch.optim.SGD(params, lr=0.01, momentum=0.9)

    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=0.001)

    if name == "adam":
        return torch.optim.Adam(params, lr=0.001)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=0.001)


def curvature_approx(model):

    total = 0
    for p in model.parameters():
        total += (p.grad**2).mean().item()

    return total


def inference_latency(model, X_test):

    start = time.time()
    with torch.no_grad():
        _ = model(X_test)
    return time.time() - start


def run(opt_name):

    X,y = fetch_california_housing(return_X_y=True)

    Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    Xtr = torch.tensor(Xtr,dtype=torch.float32)
    Ytr = torch.tensor(Ytr,dtype=torch.float32).view(-1,1)

    Xte = torch.tensor(Xte,dtype=torch.float32)
    Yte = torch.tensor(Yte,dtype=torch.float32).view(-1,1)

    model = nn.Sequential(
        nn.Linear(8,64),
        nn.ReLU(),
        nn.Linear(64,1)
    )

    opt = get_optimizer(opt_name, model.parameters())
    loss_fn = nn.MSELoss()

    grad_norms=[]
    curvatures=[]
    losses=[]

    start=time.time()

    for epoch in range(50):

        pred = model(Xtr)
        loss = loss_fn(pred,Ytr)

        opt.zero_grad()
        loss.backward()

        gn = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters())).item()
        grad_norms.append(gn)

        curvatures.append(curvature_approx(model))
        losses.append(loss.item())

        opt.step()

    train_time=time.time()-start

    latency = inference_latency(model,Xte)

    with torch.no_grad():
        pred = model(Xte).numpy()

    rmse = mean_squared_error(Yte.numpy(),pred) ** 0.5
    r2 = r2_score(Yte.numpy(),pred)

    return rmse,r2,train_time,latency,grad_norms,curvatures,losses


if __name__=="__main__":

    optimizers=["sgd","momentum","rmsprop","adam","adamw"]

    for opt in optimizers:

        rmse,r2,t,lat,gn,curv,losses = run(opt)

        print(opt,rmse,r2,t,lat)

        plt.plot(losses,label=opt)

    plt.legend()
    plt.title("Loss Curves")
    plt.show()
