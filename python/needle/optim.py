"""Optimization module"""
from collections import defaultdict
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            paramU = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * grad
            p2 = param.data - self.lr * paramU.data
            param = param.data - self.lr * paramU.data
            self.u[param] = paramU
        ### END YOUR SOLUTION

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            m, v = self.m.get(param, 0), self.v.get(param, 0)
            grad = param.grad.data + self.weight_decay * param.data
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            mHat = m/(1 - self.beta1 ** self.t)
            vHat = v/(1 - self.beta2 ** self.t)
            param.data -= self.lr * mHat.data/(vHat.data ** 0.5 + self.eps)

            self.m[param]=m
            self.v[param]=v
        ### END YOUR SOLUTION
