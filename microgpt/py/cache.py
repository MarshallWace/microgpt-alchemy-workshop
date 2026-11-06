from microgpt.py.autograd import Value

class KVCacheLayer:
    def __init__(self):
        self.keys: list[list[Value]] = []
        self.values: list[list[Value]] = []
    
    def update(self, key: list[Value], value: list[Value]) -> tuple[list[list[Value]], list[list[Value]]]:
        self.keys.append(key)
        self.values.append(value)
        return self.keys, self.values


class KVCache:
    def __init__(self):
        self.layers: list[KVCacheLayer] = []
    
    def update(self, key: list[Value], value: list[Value],
               layer_idx: int) -> tuple[list[list[Value]], list[list[Value]]]:
        while layer_idx >= len(self.layers):
            self.layers.append(KVCacheLayer())
        return self.layers[layer_idx].update(key, value)
