# What every backend's optimisers share
from typing import Any


def leaves(x: Any) -> list:
    # The weights in nested lists: a python model's matrices are lists of rows of Values
    return [leaf for item in x for leaf in leaves(item)] if isinstance(x, list) else [x]


class Optimizer:
    # torch.optim.Optimizer's interface: the weights are in param_groups, each a dict of its params and settings (lr,
    # weight_decay, ...), and state holds each weight's running averages, by the weight's id()
    def __init__(self, params: list, defaults: dict):
        groups = params if params and isinstance(params[0], dict) else [{'params': params}]
        self.param_groups = [{**defaults, **group} for group in groups]
        self.state: dict = {}

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        # Each weight's state, in the order of param_groups, and each group's settings: what a step checkpoint saves
        params = [p for group in self.param_groups for p in group['params']]
        return {'state': [self.state.get(id(p)) for p in params],
                'param_groups': [{k: v for k, v in group.items() if k != 'params'} for group in self.param_groups]}

    def load_state_dict(self, state: dict) -> None:
        params = [p for group in self.param_groups for p in group['params']]
        self.state = {id(p): saved for p, saved in zip(params, state['state']) if saved is not None}
        for group, saved in zip(self.param_groups, state['param_groups']):
            group.update(saved)

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in leaves(group['params']):
                p.grad = 0
