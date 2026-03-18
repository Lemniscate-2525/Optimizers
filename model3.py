# synthetic_optimizer_benchmark.py

import torch
import time
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cpu"

def loss_fn(theta):
    x, y = theta[0], theta[1]
    return x**2 + 100*y**2


def curvature_approx(theta):
    x, y = theta[0].item(), theta[1].item()
    return 2 + 200   # trace of Hessian


def get_optimizer(name, params):

    if name == "gd":
        return torch.optim.SGD(params, lr=0.05)

    if name == "momentum":
        return torch.optim.SGD(params, lr=0.05, momentum=0.9)

    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=0.01)

    if name == "adam":
        return torch.optim.Adam(params, lr=0.1)


def run_optimizer(opt_name):

    theta = torch.tensor([5.0, 5.0], requires_grad=True)

    opt = get_optimizer(opt_name, [theta])

    path = []
    grad_norms = []
    curvatures = []

    start = time.time()

    for epoch in range(200):

        opt.zero_grad()
        loss = loss_fn(theta)
        loss.backward()

        grad_norm = theta.grad.norm().item()
        grad_norms.append(grad_norm)

        curvatures.append(curvature_approx(theta))

        opt.step()

        path.append(theta.detach().clone().numpy())

    train_time = time.time() - start

    return np.array(path), grad_norms, curvatures, loss.item(), train_time


def plot_surface(paths_dict):

    x = np.linspace(-6,6,200)
    y = np.linspace(-6,6,200)
    X,Y = np.meshgrid(x,y)
    Z = X**2 + 100*Y**2

    plt.contour(X,Y,Z,levels=50)

    for name,path in paths_dict.items():
        plt.plot(path[:,0], path[:,1], label=name)

    plt.legend()
    plt.title("Optimizer Trajectories")
    plt.show()


if __name__ == "__main__":

    optimizers = ["gd","momentum","rmsprop","adam"]

    paths_dict = {}

    for opt in optimizers:

        path, grad_norms, curv, final_loss, t = run_optimizer(opt)

        print(opt, final_loss, t)

        paths_dict[opt] = path

        plt.plot(grad_norms,label=opt)

    plt.title("Gradient Norm per Epoch")
    plt.legend()
    plt.show()

    plot_surface(paths_dict)
