import numpy as np
from collections import OrderedDict
import math
import random
import gym
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

    def continuous_sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage) - 4, size=int(batch_size / 4))
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            for j in range(4):
                X, Y, U, R, D = self.storage[i + j]
                x.append(np.array(X, copy=False))
                y.append(np.array(Y, copy=False))
                u.append(np.array(U, copy=False))
                r.append(np.array(R, copy=False))
                d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())

    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr * self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]

    def restore(self):
        self.update(lr=0)