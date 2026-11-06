# Adapted from Andrej Karpathy's microgpt (MIT licence; see THIRD_PARTY_NOTICES.md)
from microgpt.py.autograd import Value
import random

def matrix(nout: int, nin: int, std: float = 0.08) -> list[list[Value]]:  # -> [nout, nin]
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x: list[Value]) -> list[Value]:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def log_softmax(logits: list[Value]) -> list[Value]:
    # log(softmax(x)) computed without log(0) when a probability underflows
    max_val = max(val.data for val in logits)
    log_total = sum((val - max_val).exp() for val in logits).log()
    return [val - max_val - log_total for val in logits]

def cross_entropy(logits: list[list[Value]], targets: list[int]) -> Value:
    losses = [-log_softmax(lg)[t] for lg, t in zip(logits, targets)]
    return sum(losses) / len(losses)

def scaled_dot_product_attention(q_h: list[Value], k_h: list[list[Value]], v_h: list[list[Value]]) -> list[Value]:
    # q_h: [head_dim] for the current token, k_h/v_h: [seq_len, head_dim] for all tokens so far
    head_dim = len(q_h)
    attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
    attn_weights = softmax(attn_logits)
    head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
    return head_out
