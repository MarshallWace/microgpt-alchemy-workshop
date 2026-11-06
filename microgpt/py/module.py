# The base class of every layer: finds its weights by name, and saves and loads them
from microgpt.py.autograd import Value


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix: str = '') -> dict[str, list[list[Value]]]:
        # Every weight matrix by name, such as 'blocks.0.attn.wq.weight'; a tied matrix once, under its first name
        named = []
        for name, attr in vars(self).items():
            if isinstance(attr, Module):
                named += attr.named_parameters(f'{prefix}{name}.').items()
            elif isinstance(attr, list) and attr and isinstance(attr[0], Module):
                for i, layer in enumerate(attr):
                    named += layer.named_parameters(f'{prefix}{name}.{i}.').items()
            elif isinstance(attr, list) and attr and isinstance(attr[0], list):
                named.append((prefix + name, attr))
        unique = {}
        for name, m in named:
            unique.setdefault(id(m), (name, m))
        return dict(unique.values())

    def parameters(self) -> list[Value]:
        return [p for m in self.named_parameters().values() for row in m for p in row]

    def state_dict(self) -> dict[str, list[list[float]]]:
        return {name: [[p.data for p in row] for row in m] for name, m in self.named_parameters().items()}

    def load_state_dict(self, weights: dict) -> None:
        matrices = self.named_parameters()
        assert weights.keys() == matrices.keys(), f'expected weights {sorted(matrices)}, got {sorted(weights)}'
        for name, m in matrices.items():
            for row, values in zip(m, weights[name]):
                for p, w in zip(row, values):
                    p.data = float(w)
