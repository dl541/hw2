"""The module.
"""
from functools import reduce
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(fan_in=in_features, fan_out=out_features)
        )
        self.bias = (
            Parameter(
                init.kaiming_uniform(fan_in=out_features, fan_out=1).reshape(
                    (1, out_features)
                )
            )
            if bias
            else None
        )
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.bias != None:
            temp = X @ self.weight
            return temp + ops.broadcast_to(self.bias, temp.shape)
        else:
            return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batchSize = X.shape[0]
        return ops.reshape(X, (batchSize, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return reduce(lambda u, m: m.forward(u), self.modules, x)
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batchSize, classes = logits.shape
        logsumexp = ops.logsumexp(logits, axes=(1,))
        Zy = ops.summation(init.one_hot(classes, y) * logits, axes=(1,))
        return ops.summation(logsumexp - Zy) / batchSize
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(np.ones((dim)), device=device, dtype=dtype)
        self.bias = Parameter(np.zeros((dim)), device=device, dtype=dtype)
        self.running_mean = Tensor(
            np.zeros((dim)), device=device, dtype=dtype, requires_grad=False
        )
        self.running_var = Tensor(
            np.ones((dim)), device=device, dtype=dtype, requires_grad=False
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, n_feature = x.shape[0], x.shape[1]
        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        if self.training:
            m = x.sum(0) / n
            mean = ops.broadcast_to(m, x.shape)
            v = ((x - mean) ** 2).sum(0) / n
            var = ops.broadcast_to(v, x.shape)
            ret = w * (x - mean) / ((var + self.eps) ** 0.5) + b

            self.running_mean = (
                self.momentum * m.data + (1 - self.momentum) * self.running_mean.data
            )
            self.running_var = (
                self.momentum * v.data + (1 - self.momentum) * self.running_var.data
            )
            return ret
        else:
            mean = ops.broadcast_to(self.running_mean, x.shape)
            var = ops.broadcast_to(self.running_var, x.shape)
            ret = (x - mean) / ((var + self.eps) ** 0.5) * w + b
            return ret
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n = x.shape[0]
        mean = ops.broadcast_to((x.sum(1) / self.dim).reshape((n, 1)), x.shape)
        var = ((x - mean) ** 2 / self.dim).sum(1).reshape((n, 1))
        ret = (x - mean) / ops.broadcast_to(
            ops.power_scalar(var + self.eps, 0.5), x.shape
        ) * ops.broadcast_to(self.weight, x.shape) + ops.broadcast_to(
            self.bias, x.shape
        )

        return ret
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * init.randb(*x.shape, p=self.p) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
