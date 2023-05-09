import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ReLU_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    temp = b / torch.sqrt(a * c)
    temp = temp.to(device)
    temp = torch.minimum(temp, torch.tensor(1))  # Need the min due to numerical errors
    kappa = torch.maximum(temp, -torch.tensor(1))
    return (torch.sqrt(1 - kappa ** 2) + (torch.pi - torch.arccos(kappa) * kappa)) * torch.sqrt(a * c) / (2 * torch.pi)


def erf_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return (2 / np.pi) * np.arcsin(b / (np.sqrt((a + 0.5) * (c + 0.5))))


def exp_phi(a, b, c, sigma=4.):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return np.exp((a + 2 * b + c) / (2 * sigma ** 2))


def id_phi(a, b, c):
    # a, b, c = K_{t,t}, K_{t,s}, K_{s,s}
    return b
