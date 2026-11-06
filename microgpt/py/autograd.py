# From Andrej Karpathy's microgpt, with type hints added (MIT licence; see THIRD_PARTY_NOTICES.md)
from __future__ import annotations

import math

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data: float, children: tuple[Value, ...] = (), local_grads: tuple[float, ...] = ()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: float) -> Value: return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self) -> Value: return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self) -> Value: return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self) -> Value: return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self) -> Value: return self * -1
    def __radd__(self, other: float) -> Value: return self + other
    def __sub__(self, other: Value | float) -> Value: return self + (-other)
    def __rsub__(self, other: float) -> Value: return other + (-self)
    def __rmul__(self, other: float) -> Value: return self * other
    def __truediv__(self, other: Value | float) -> Value: return self * other**-1
    def __rtruediv__(self, other: float) -> Value: return other * self**-1

    def backward(self) -> None:
        topo = []
        visited = set()
        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad