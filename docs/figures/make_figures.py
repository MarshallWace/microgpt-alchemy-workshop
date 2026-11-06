# Draws the figures used in docs/how-a-gpt-works.md and README.md as SVG files in this folder. Standard library only.
# Run from the repository root: python3 docs/figures/make_figures.py
import html
import math
import os

ACCENT, GREY, INK = '#a152c2', '#8a8a8a', '#1a1a1a'
BOX, BOX_GREY = '#efe3f5', '#e9e9e9'  # opaque fills, so text stays readable on light and dark pages
FONT = 'Lato, Calibri, Segoe UI, Arial, sans-serif'
MONO = 'Roboto Mono, Consolas, monospace'
DARK = {'ink': '#ececec', 'box': '#3e2a4a', 'boxg': '#333333'}  # colours on a dark page
RULES = ''.join(f'.{k} {{fill: {v}}}' for k, v in DARK.items()) + '.grid {stroke: #333333}'
STYLE = '@media (prefers-color-scheme: dark) {' + RULES + '}'
OUT = os.path.dirname(os.path.abspath(__file__))


def shade(v: float) -> str:
    # Solid colour between white (v = 0) and the accent (v = 1)
    t = 0.12 + 0.88 * v
    return '#%02x%02x%02x' % tuple(round(255 + (c - 255) * t) for c in (161, 82, 194))


def svg(name: str, w: int, h: int, parts: list[str]) -> None:
    defs = ''.join(f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>' for key, colour in (('g', GREY), ('a', ACCENT)))
    body = '\n'.join(parts)
    for cls, colour in (('ink', INK), ('box', BOX), ('boxg', BOX_GREY)):
        body = body.replace(f'fill="{colour}"', f'class="{cls}" fill="{colour}"')
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                f'font-family="{FONT}"><style>{STYLE}</style><defs>{defs}</defs>\n{body}\n</svg>\n')


def text(x, y, s, size=16, fill=GREY, anchor='middle', mono=False, bold=False) -> str:
    extra = (f' font-family="{MONO}"' if mono else '') + (' font-weight="700"' if bold else '')
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>{html.escape(s)}</text>'


def rect(x, y, w, h, fill=BOX, rx=8, stroke=None, width=2) -> str:
    border = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ''
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{border}/>'


def box(x, y, w, h, label, sub=None, fill=BOX, size=18, mono=False) -> str:
    cy = y + h / 2 + (size * 0.35 if sub is None else -2)
    out = rect(x, y, w, h, fill) + text(x + w / 2, cy, label, size, INK, mono=mono)
    return out + (text(x + w / 2, y + h / 2 + 18, sub, 13, GREY) if sub else '')


def arrow(x1, y1, x2, y2, colour=GREY, width=2, dashed=False) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    marker = 'a' if colour == ACCENT else 'g'
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="{width}"'
            f'{dash} marker-end="url(#{marker})"/>')


def path(d, colour=GREY, width=2, dashed=False, end=True) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    marker = f' marker-end="url(#{"a" if colour == ACCENT else "g"})"' if end else ''
    return f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{dash}{marker}/>'


def chip(x, y, label, w=44, h=44, fill=BOX, size=20, colour=INK) -> str:
    return rect(x, y, w, h, fill, rx=6) + text(x + w / 2, y + h / 2 + size * 0.35, label, size, colour, mono=True)


def vector(x, y, values, cell=15, h=26) -> str:
    # A vector drawn as a strip of shaded cells
    return ''.join(rect(x + i * cell, y, cell - 2, h, shade(v), rx=2) for i, v in enumerate(values))


def wave(n, seed):
    return [0.5 + 0.45 * math.sin(seed * 1.7 + i * 2.3) * math.cos(seed + i * 0.9) for i in range(n)]


def overview():
    p = [chip(110, 10, '"good"', w=100, h=40, size=18, colour=ACCENT)]
    rows = [('tokenizer', 'text → numbers', 74), ('embeddings', 'numbers → vectors that can carry meaning', 154)]
    for label, desc, y in rows:
        p += [box(50, y, 220, 50, label), text(300, y + 31, desc, 16, GREY, 'start')]
    p += [rect(36, 234, 248, 150, 'none', stroke=ACCENT), text(160, 256, 'transformer block × layers', 15, ACCENT, bold=True),
          box(50, 266, 220, 46, 'attention'), text(300, 295, 'each character looks at the earlier ones', 16, GREY, 'start'),
          box(50, 324, 220, 46, 'MLP'), text(300, 353, 'each character is processed on its own', 16, GREY, 'start'),
          box(50, 404, 220, 50, 'output layer'), text(300, 435, 'a score for every possible next character', 16, GREY, 'start')]
    for y1, y2 in [(50, 74), (124, 154), (204, 234), (384, 404), (454, 478)]:
        p.append(arrow(160, y1, 160, y2 - 2))
    p.append(text(160, 500, 'scores for the next character', 16, INK))
    svg('overview.svg', 680, 520, p)


def tokens():
    p = [text(40, 42, 'text', 15, GREY, 'start'), text(40, 122, 'tokens', 15, GREY, 'start')]
    for i, (ch, n) in enumerate(zip('good', [61, 69, 69, 58])):
        x = 130 + i * 70
        p += [chip(x, 16, ch, w=52, h=44), arrow(x + 26, 62, x + 26, 92), chip(x, 96, str(n), w=52, h=44, fill=BOX_GREY, size=18)]
    svg('tokens.svg', 440, 156, p)


def embeddings():
    token, pos = wave(16, 1), wave(16, 4)
    total = [min(1, max(0, (a + b) / 2)) for a, b in zip(token, pos)]
    p = [chip(20, 18, 'o', w=44, h=36), text(80, 42, 'token', 15, GREY, 'start'), vector(170, 22, token),
         text(420, 41, 'vector of "o"', 15, GREY, 'start'),
         chip(20, 74, '2', w=44, h=36, fill=BOX_GREY), text(80, 98, 'position', 15, GREY, 'start'), vector(170, 78, pos),
         text(420, 97, 'vector of position 2', 15, GREY, 'start'),
         text(150, 130, '+', 26, GREY), f'<line x1="170" y1="122" x2="408" y2="122" stroke="{GREY}" stroke-width="1.5"/>',
         vector(170, 136, total), text(420, 155, '"o" in third place', 15, INK, 'start', bold=True),
         text(170, 196, '16 numbers per vector, learned during training', 14, GREY, 'start')]
    svg('embeddings.svg', 620, 210, p)


def qkv():
    p = [box(20, 82, 150, 52, 'token vector')]
    for i, (name, question) in enumerate([('query', 'what am I looking for?'), ('key', 'what do I contain?'),
                                          ('value', 'what do I give?')]):
        y = 14 + i * 70
        p += [path(f'M170,108 C210,108 210,{y + 24} 248,{y + 24}'), box(252, y, 120, 48, name),
              text(392, y + 30, question, 16, GREY, 'start')]
    svg('qkv.svg', 620, 222, p)


def kv_cache():
    words = ['the', 'film', 'was']
    p = [text(30, 24, 'reads', 15, GREY, 'start'), text(210, 24, 'cache: keys and values of everything read so far', 15, GREY, 'start')]
    for i, word in enumerate(words):
        y = 40 + i * 58
        p += [chip(30, y, word, w=80, size=18, colour=ACCENT), arrow(118, y + 22, 196, y + 22)]
        for j in range(i + 1):
            p.append(chip(210 + j * 92, y, words[j], w=80, size=18, fill=BOX if j == i else BOX_GREY))
    svg('kv_cache.svg', 600, 216, p)


def mlp():
    small = wave(16, 2)
    steps = [(small, 10, '16 numbers'), ('linear', None), (wave(64, 3), 3, '64 numbers'), ('relu', 'negatives → 0'),
             ('linear', None), (small[::-1], 10, '16 numbers')]
    p, x = [], 10
    for i, step in enumerate(steps):
        if i:
            p.append(arrow(x + 4, 41, x + 30, 41))
            x += 36
        if isinstance(step[0], str):
            p.append(box(x, 18, 84, 46, step[0], size=18) + (text(x + 42, 86, step[1], 16, GREY) if step[1] else ''))
            x += 84
        else:
            values, cell, label = step
            p += [vector(x, 26, values, cell=cell, h=30), text(x + len(values) * cell / 2, 86, label, 16, GREY)]
            x += len(values) * cell
    svg('mlp.svg', x + 10, 100, p)

def block():
    p = [text(30, 112, 'x', 22, INK, mono=True), arrow(48, 106, 74, 106),
         box(78, 82, 110, 48, 'RMSNorm', size=16), arrow(190, 106, 218, 106), box(222, 82, 120, 48, 'attention', size=16),
         arrow(344, 106, 376, 106), f'<circle cx="398" cy="106" r="20" fill="{BOX}"/>', text(398, 113, '+', 22, INK),
         arrow(418, 106, 446, 106), box(450, 82, 110, 48, 'RMSNorm', size=16), arrow(562, 106, 590, 106),
         box(594, 82, 100, 48, 'MLP', size=16), arrow(696, 106, 728, 106), f'<circle cx="750" cy="106" r="20" fill="{BOX}"/>',
         text(750, 113, '+', 22, INK), arrow(770, 106, 810, 106), text(850, 112, 'out', 18, INK, mono=True),
         path('M60,106 L60,40 L398,40 L398,84', ACCENT, 2.5), path('M432,106 L432,40 L750,40 L750,84', ACCENT, 2.5),
         text(229, 32, 'copy of x, added back', 14, ACCENT), text(591, 32, 'copy, added back', 14, ACCENT),
         text(440, 170, 'each part adds to the vector instead of replacing it', 15, GREY)]
    svg('block.svg', 900, 186, p)


def softmax_fig():
    toks, logits = ['d', 's', 'e'], [2.0, 1.0, 0.1]
    e = [math.exp(x) for x in logits]
    probs = [x / sum(e) for x in e]
    p = [f'<text x="320" y="30" font-size="17" fill="{GREY}" text-anchor="middle">the model has read '
         f'<tspan font-family="{MONO}" fill="{ACCENT}" font-weight="700">goo</tspan> — what comes next?</text>',
         text(150, 74, 'score (logit)', 15), text(490, 74, 'probability', 15)]
    for i, (t, logit, prob) in enumerate(zip(toks, logits, probs)):
        y = 96 + i * 56
        p += [text(40, y + 25, t, 24, ACCENT, mono=True, bold=True),
              rect(70, y + 8, max(45 * logit, 2), 24, GREY, rx=4), text(70 + max(45 * logit, 2) + 8, y + 26, f'{logit:.1f}', 17, GREY, 'start'),
              rect(400, y + 8, 190 * prob, 24, ACCENT, rx=4), text(400 + 190 * prob + 8, y + 26, f'{prob:.0%}', 17, GREY, 'start')]
    p += [arrow(245, 172, 370, 172), text(307, 162, 'softmax', 16),
          text(320, 290, 'any number in, positive numbers that add up to 100% out · the real model scores all 96 tokens', 14)]
    svg('softmax.svg', 640, 304, p)


def attention_matrix():
    toks = ['the', 'film', 'was', 'not', 'good']
    w = [[1.00], [0.35, 0.65], [0.20, 0.50, 0.30], [0.15, 0.20, 0.35, 0.30], [0.15, 0.10, 0.12, 0.50, 0.13]]
    cell, left, top = 78, 150, 120
    p = [text(left + cell * 2.5, 30, 'looks at →', 16),
         f'<text x="30" y="{top + cell * 2.5}" font-size="16" fill="{GREY}" text-anchor="middle" '
         f'transform="rotate(-90 30 {top + cell * 2.5})">token doing the looking →</text>']
    for j, t in enumerate(toks):
        p.append(text(left + cell * j + cell / 2, top - 16, t, 18, GREY, mono=True))
    for i, t in enumerate(toks):
        p.append(text(left - 16, top + cell * i + cell / 2 + 6, t, 18, ACCENT if t == 'good' else GREY, 'end', mono=True, bold=t == 'good'))
        for j in range(5):
            x, y = left + cell * j + 3, top + cell * i + 3
            if j <= i:
                v = w[i][j]
                p += [rect(x, y, cell - 6, cell - 6, shade(v), rx=6),
                      text(x + (cell - 6) / 2, y + (cell - 6) / 2 + 7, f'{v:.2f}', 19, '#ffffff' if v >= 0.6 else '#222222')]
            else:
                p.append(rect(x, y, cell - 6, cell - 6, BOX_GREY, rx=6))
    p += [rect(left, top + cell * 4, cell * 5, cell, 'none', stroke=ACCENT, width=3),
          text(left + cell * 2.5, top + cell * 5 + 34, 'each row adds up to 1 · grey: later tokens, which cannot be seen', 15)]
    svg('attention_matrix.svg', left + cell * 5 + 90, top + cell * 5 + 70, p)


def loss():
    rows = [(1.00, 'sure and right'), (0.50, ''), (0.10, ''), (0.01, 'confident and wrong')]
    p = [text(30, 24, 'probability given to the correct token', 15, GREY, 'start'), text(470, 24, 'loss = −log(p)', 15, GREY, 'start')]
    for i, (prob, note) in enumerate(rows):
        y = 40 + i * 44
        p += [text(70, y + 22, f'{prob:.2f}', 17, INK, 'end'), rect(84, y + 4, max(340 * prob, 2), 26, ACCENT, rx=4),
              text(470, y + 23, f"{abs(math.log(prob)):.2f}", 18, INK, 'start', bold=True), text(530, y + 23, note, 14, GREY, 'start')]
    svg('loss.svg', 700, 222, p)


def loss_curve():
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # The starter's real loss: the mean of each 10 steps, as train.py logs it
    values = [4.1537, 3.1723, 3.0511, 3.0772, 2.8410, 2.7624, 2.8723, 2.6965, 2.8233, 2.8248]
    x0, x1, y0, y1, lo, hi = 70, 560, 250, 30, 2.5, 4.8
    X = lambda s: x0 + (s - 1) / 99 * (x1 - x0)
    Y = lambda v: y0 - (v - lo) / (hi - lo) * (y0 - y1)
    p = []
    for v in (2.5, 3.0, 3.5, 4.0, 4.5):
        p += [f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x1}" y2="{Y(v):.1f}" class="grid" stroke="{BOX_GREY}" stroke-width="1"/>',
              text(x0 - 10, Y(v) + 5, f'{v:.1f}', 13, GREY, 'end')]
    for s in (1, 25, 50, 75, 100):
        p.append(text(X(s), y0 + 22, str(s), 13, GREY))
    p += [text((x0 + x1) / 2, y0 + 44, 'training step', 14), text(20, y1 - 10, 'loss', 14, GREY, 'start'),
          f'<line x1="{x0}" y1="{Y(4.56):.1f}" x2="{x1}" y2="{Y(4.56):.1f}" stroke="{GREY}" stroke-width="1.5" stroke-dasharray="6 5"/>',
          text(x1, Y(4.56) - 8, 'knows nothing: log(96) = 4.56', 13, GREY, 'end'),
          path('M' + ' L'.join(f'{X(s):.1f},{Y(v):.1f}' for s, v in zip(steps, values)), ACCENT, 2.5, end=False)]
    p += [f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="{ACCENT}"/>' for s, v in zip(steps, values)]
    p += [text(X(100) + 8, Y(2.82) + 5, '2.82', 15, ACCENT, 'start', bold=True), text(X(10) + 10, Y(4.15) - 4, '4.15', 15, ACCENT, 'start', bold=True)]
    svg('loss_curve.svg', 640, 310, p)


def autograd():
    def value(x, y, name, v, grad):
        return (rect(x, y, 96, 46, BOX) + text(x + 48, y + 30, f'{name} = {v}', 18, INK, mono=True) +
                text(x + 48, y + 70, f'grad {grad}', 16, ACCENT, bold=True))
    def op(x, y, sym):
        return f'<circle cx="{x}" cy="{y}" r="22" fill="{BOX_GREY}"/>' + text(x, y + 7, sym, 22, INK)
    p = [value(20, 30, 'a', '2', '−2'), value(20, 170, 'b', '−3', '2'), op(240, 123, '×'), value(310, 100, 'c', '−6', '1'),
         op(500, 123, '+'), value(570, 100, 'd', '−4', '1'),
         arrow(118, 64, 220, 112), arrow(118, 186, 220, 134), arrow(262, 123, 306, 123), arrow(408, 123, 476, 123),
         arrow(524, 123, 566, 123), path('M118,44 C300,20 460,40 494,100'),
         text(690, 24, 'a is used twice, so its two gradients add up: −3 + 1', 14, GREY, 'end'),
         text(20, 290, 'values: calculated forwards, first', 14, INK, 'start'),
         text(380, 290, 'grad: calculated backwards, after', 14, ACCENT, 'start')]
    svg('autograd.svg', 700, 306, p)


def training_loop():
    steps = [('guess', 'reads "the fil"', 'gives "m" 12%'), ('loss', '−log(0.12)', '= 2.12'),
             ('gradients', 'backward()', 'which way to turn each knob'), ('update', 'AdamW', 'turns every knob a little')]
    p = []
    for i, (name, line1, line2) in enumerate(steps):
        x = 20 + i * 225
        p += [rect(x, 20, 200, 110, BOX), f'<circle cx="{x + 24}" cy="44" r="14" fill="{ACCENT}"/>',
              text(x + 24, 50, str(i + 1), 16, '#ffffff', bold=True), text(x + 46, 51, name, 19, INK, 'start', bold=True),
              text(x + 100, 88, line1, 15, INK, mono=line1 != 'AdamW'), text(x + 100, 112, line2, 15, GREY)]
        if i < 3:
            p.append(arrow(x + 202, 75, x + 222, 75))
    p += [path('M795,132 L795,170 L120,170 L120,134', ACCENT, 2.5), text(457, 162, 'repeat with the next piece of text', 15, ACCENT, bold=True)]
    svg('training_loop.svg', 920, 186, p)

def chips(x, y, labels, colour=INK, fill=BOX):
    # A row of token chips; <bos> and other special tokens are grey
    return [chip(x + i * 66, y, t, w=58, size=16, colour=colour, fill=BOX_GREY if t.startswith('<') else fill)
            for i, t in enumerate(labels)]


def next_token():
    inputs, targets = ['<bos>', 't', 'h', 'e', '␣', 'f', 'i', 'l'], ['t', 'h', 'e', '␣', 'f', 'i', 'l', 'm']
    p = [text(20, 40, 'reads', 15, GREY, 'start'), text(20, 132, 'must guess', 15, GREY, 'start')]
    p += chips(120, 18, inputs) + chips(120, 110, targets, '#222222', shade(0.25))
    p += [arrow(120 + i * 66 + 29, 64, 120 + i * 66 + 29, 106) for i in range(len(inputs))]
    p.append(text(120 + 4 * 66 - 4, 186, 'every position guesses the character after it, so one text gives many examples', 15))
    svg('next_token.svg', 680, 200, p)


def probe():
    p = [box(10, 30, 140, 52, 'SST sentence', size=16), arrow(152, 56, 180, 56),
         box(184, 30, 150, 52, 'frozen model', 'weights do not change', fill=BOX_GREY, size=16), arrow(336, 56, 364, 56),
         rect(368, 30, 130, 52, BOX), vector(378, 44, wave(14, 5), cell=8, h=24), text(433, 100, 'features', 13),
         arrow(500, 56, 528, 56), box(532, 30, 170, 52, 'logistic regression', size=16), arrow(704, 56, 732, 56),
         text(740, 50, 'positive', 16, ACCENT, 'start', bold=True), text(740, 72, 'or negative?', 16, ACCENT, 'start', bold=True)]
    svg('probe.svg', 850, 112, p)


def word_vectors():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    raw = lambda x, y, z: ((x - z) * c, -y + (x + z) * s)  # isometric view: x = gender, y = royalty (up), z = depth
    pts = {'man': (0.5, 1.2, 1.4), 'woman': (2.7, 1.2, 1.4), 'king': (0.5, 3.1, 0.5), 'queen': (2.7, 3.1, 0.5)}
    axes = {'male → female': ((4.0, 0, 0), 0, 24, 'end'), 'royalty': ((0, 3.9, 0), 0, -12, 'middle'),
            'other meanings': ((0, 0, 2.6), 0, 24, 'start')}
    W, H, M = 600, 440, 50
    allp = [raw(*q) for q in pts.values()] + [raw(*a[0]) for a in axes.values()] + [raw(0, 0, 0)]
    allp += [raw(x, 0, z) for x, _, z in pts.values()]
    xs, ys = [q[0] for q in allp], [q[1] for q in allp]
    k = min((W - 2 * M) / (max(xs) - min(xs)), (H - 2 * M - 30) / (max(ys) - min(ys)))
    P = lambda x, y, z: (M + (raw(x, y, z)[0] - min(xs)) * k, M + (raw(x, y, z)[1] - min(ys)) * k)
    p = []
    for label, (end, dx, dy, anchor) in axes.items():
        (ax, ay), (bx, by) = P(0, 0, 0), P(*end)
        p += [arrow(ax, ay, bx, by, width=1.5), text(bx + dx, by + dy, label, 15, GREY, anchor)]
    for x, y, z in pts.values():  # dotted drop lines show how far back each point is
        (ax, ay), (fx, fy) = P(x, y, z), P(x, 0, z)
        p += [f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{fx:.1f}" y2="{fy:.1f}" stroke="{GREY}" stroke-dasharray="3 4" opacity="0.7"/>',
              f'<ellipse cx="{fx:.1f}" cy="{fy:.1f}" rx="5" ry="2.5" fill="{GREY}" opacity="0.5"/>']
    for a, b in [('man', 'king'), ('woman', 'queen')]:
        (ax, ay), (bx, by) = P(*pts[a]), P(*pts[b])
        p.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{GREY}" stroke-width="1.2" stroke-dasharray="6 5"/>')
    for a, b in [('man', 'woman'), ('king', 'queen')]:  # the same step, highlighted
        (ax, ay), (bx, by) = P(*pts[a]), P(*pts[b])
        f = 13 / math.hypot(bx - ax, by - ay)
        p.append(arrow(ax + (bx - ax) * f, ay + (by - ay) * f, bx - (bx - ax) * f, by - (by - ay) * f, ACCENT, 3))
    side = {'man': ('end', -14, 6), 'king': ('end', -14, 6), 'woman': ('start', 14, -2), 'queen': ('start', 14, 6)}
    for name, q in pts.items():
        (px, py), (anchor, dx, dy) = P(*q), side[name]
        p += [f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="{ACCENT}"/>', text(px + dx, py + dy, name, 18, ACCENT, anchor, bold=True)]
    p.append(text(W / 2, H - 14, 'man → woman is the same step as king → queen', 15))
    svg('word_vectors.svg', W, H, p)


if __name__ == '__main__':
    for draw in (overview, tokens, embeddings, word_vectors, qkv, attention_matrix, kv_cache, mlp, block,
                 softmax_fig, loss, loss_curve, autograd, training_loop, next_token, probe):
        draw()
    print(f"wrote {len([f for f in os.listdir(OUT) if f.endswith('.svg')])} figures to {OUT}")
