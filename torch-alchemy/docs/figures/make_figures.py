# Draws the guide's diagrams (docs/*.md) as SVG files in this folder. Standard library only.
# Run from the repository root: python3 docs/figures/make_figures.py
import html
import os

ACCENT, GREY, INK = '#a152c2', '#8a8a8a', '#1a1a1a'
# Opaque fills, so text stays readable on light and dark pages
BOX, BOX_GREY, ACCENT_BOX = '#efe3f5', '#e9e9e9', '#dcc1ea'
FONT = 'Lato, Calibri, Segoe UI, Arial, sans-serif'
MONO = 'Roboto Mono, Consolas, monospace'
DARK = {'ink': '#ececec', 'box': '#3e2a4a', 'boxg': '#333333', 'boxa': '#5c3a70'}  # colours on a dark page
RULES = ''.join(f'.{k} {{fill: {v}}}' for k, v in DARK.items())
STYLE = '@media (prefers-color-scheme: dark) {' + RULES + '}'
OUT = os.path.dirname(os.path.abspath(__file__))


def shade(v: float) -> str:
    # Solid colour between white (v = 0) and the accent (v = 1)
    t = 0.12 + 0.88 * v
    return '#%02x%02x%02x' % tuple(round(255 + (c - 255) * t) for c in (161, 82, 194))


def svg(name: str, w: int, h: int, parts: list[str]) -> None:
    defs = ''.join(f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                   f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{colour}"/></marker>'
                   for key, colour in (('g', GREY), ('a', ACCENT)))
    body = '\n'.join(parts)
    for cls, colour in (('ink', INK), ('box', BOX), ('boxg', BOX_GREY), ('boxa', ACCENT_BOX)):
        body = body.replace(f'fill="{colour}"', f'class="{cls}" fill="{colour}"')
    with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                f'font-family="{FONT}"><style>{STYLE}</style><defs>{defs}</defs>\n{body}\n</svg>\n')


def text(x, y, s, size=16, fill=GREY, anchor='middle', mono=False, bold=False) -> str:
    extra = (f' font-family="{MONO}"' if mono else '') + (' font-weight="700"' if bold else '')
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{extra}>'
            f'{html.escape(s)}</text>')


def rect(x, y, w, h, fill=BOX, rx=8, stroke=None) -> str:
    border = f' stroke="{stroke}" stroke-width="2"' if stroke else ''
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{border}/>'


def box(x, y, w, h, label, sub=None, fill=BOX, size=16, mono=False) -> str:
    cy = y + h / 2 + (size * 0.35 if sub is None else -2)
    out = rect(x, y, w, h, fill) + text(x + w / 2, cy, label, size, INK, mono=mono)
    return out + (text(x + w / 2, y + h / 2 + 17, sub, 12) if sub else '')


def arrow(x1, y1, x2, y2, colour=GREY) -> str:
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colour}" stroke-width="2" '
            f'marker-end="url(#{"a" if colour == ACCENT else "g"})"/>')


def vector(x, y, values, cell=9, h=22) -> list[str]:
    # A vector as a row of bars, darker for bigger values
    return [rect(x + i * cell, y, cell - 2, h, shade(v), rx=2) for i, v in enumerate(values)]


WAVE = [0.9, 0.2, 0.6, 0.7, 0.1, 0.5, 0.3, 0.8, 0.4, 1.0, 0.2, 0.6]


def step(y, number, label, sub=None) -> list[str]:
    # A numbered step's label on the left of a diagram that reads from top to bottom
    out = [text(20, y, number, 15, ACCENT, 'start', bold=True), text(40, y, label, 14, INK, 'start')]
    return out + ([text(40, y + 17, sub, 12, anchor='start')] if sub else [])


def chips(x, y, labels, w=60, h=30, gap=4, fill=BOX, size=13, grey=()) -> list[str]:
    # A row of small boxes (tokens, windows, files); labels in grey are drawn grey
    return [box(x + i * (w + gap), y, w, h, label, fill=BOX_GREY if label in grey else fill, size=size, mono=True)
            for i, label in enumerate(labels)]


def bracket(x1, x2, y, label=None, up=False) -> list[str]:
    # A bracket under (or over) x1..x2, with an optional label beneath it
    d = -8 if up else 8
    out = [f'<path d="M{x1} {y} v{d} H{x2} v{-d}" fill="none" stroke="{GREY}" stroke-width="2"/>']
    return out + ([text((x1 + x2) / 2, y + (2 * d if up else 2 * d + 8), label, 12)] if label else [])


def split(x_from, y, xs, y_to) -> list[str]:
    # One arrow fanning out to several boxes below: down, along, and down to each
    mid = y + (y_to - y) / 2
    out = [f'<path d="M{x_from} {y} V{mid}" stroke="{GREY}" stroke-width="2"/>',
           f'<path d="M{min(xs)} {mid} H{max(xs)}" stroke="{GREY}" stroke-width="2"/>']
    return out + [arrow(x, mid, x, y_to) for x in xs]


def pipeline() -> None:
    p = [box(10, 30, 150, 56, '"a gripping film"', 'an SST sentence'), arrow(162, 58, 190, 58),
         box(194, 30, 170, 56, 'frozen model', 'trained only on next tokens', fill=BOX_GREY), arrow(366, 58, 394, 58),
         rect(398, 30, 130, 56, BOX), *vector(409, 47, WAVE), text(463, 106, 'features', 13),
         arrow(530, 58, 558, 58), box(562, 30, 180, 56, 'logistic regression', 'the only part that learns'),
         arrow(744, 58, 772, 58), text(780, 54, 'positive', 16, ACCENT, 'start', bold=True),
         text(780, 76, 'or negative?', 16, ACCENT, 'start', bold=True)]
    svg('probe_pipeline.svg', 890, 118, p)


def features() -> None:
    # Numbered steps from top to bottom, each labelled on the left
    tokens = ['<bos>', ' a', ' gri', 'pping', ' film']
    x0, w, n = 250, 88, len(tokens)
    centre = [x0 + i * w + 40 for i in range(n)]
    last_x = centre[-1]

    p = step(43, '1', 'tokenise, <bos> first')
    p += [box(x0 + i * w, 20, 80, 36, t, fill=BOX_GREY if i == 0 else BOX, size=15, mono=True)
          for i, t in enumerate(tokens)]
    p += [arrow(c, 58, c, 80) for c in centre]
    p += step(107, '2', 'frozen model, no output layer')
    p += [box(x0, 84, n * w - 8, 36, 'last layer of model.model', fill=BOX_GREY, size=15)]
    p += [arrow(c, 122, c, 144) for c in centre]
    p += step(163, '3', 'one vector per token')
    for c in centre:
        p += vector(c - 36, 148, WAVE[int(c) % 5:int(c) % 5 + 8])
    p += step(239, '4', "the last token's vector")
    p += [arrow(last_x, 172, last_x, 214), rect(last_x - 44, 218, 88, 36, BOX), *vector(last_x - 36, 225, WAVE[3:11]),
          text(last_x, 274, 'the feature vector: hidden_dim numbers', 12, anchor='end'),
          text(x0 + 2 * w + 40, 205, 'the only position that has read the whole text', 12)]
    svg('probe_features.svg', 700, 290, p)


def fit() -> None:
    cs = ['1/64', '1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4']
    dev = [0.78, 0.80, 0.815, 0.822, 0.818, 0.81, 0.80, 0.795, 0.79]  # an illustration, not real numbers
    best, base, top, x0 = dev.index(max(dev)), 200, 80, 110
    p = [text(20, 28, '1', 15, ACCENT, 'start', bold=True),
         text(40, 28, 'fit a probe on train for each C, and measure it on dev', 14, INK, 'start')]
    for i, (c, d) in enumerate(zip(cs, dev)):
        h = (d - 0.74) / 0.1 * (base - top)
        p += [rect(x0 + i * 50, base - h, 36, h, ACCENT if i == best else BOX, rx=3),
              text(x0 + 18 + i * 50, base + 18, c, 12, mono=True)]
    p += [text(x0 - 12, 142, 'dev', 13, anchor='end'), text(x0 - 12, 158, 'accuracy', 13, anchor='end'),
          text(x0 + 18 + best * 50, 72, 'best', 12, ACCENT, bold=True),
          arrow(x0 + 180, base + 38, x0 - 4, base + 38), text(x0 + 186, base + 42, 'C', 13, INK),
          arrow(x0 + 206, base + 38, x0 + 440, base + 38),
          text(x0, base + 58, 'few features, simple probe', 12, anchor='start'),
          text(x0 + 440, base + 58, 'many features, can memorise train', 12, anchor='end'),
          text(600, 28, '2', 15, ACCENT, 'start', bold=True), text(620, 28, 'keep the best one', 14, INK, 'start'),
          arrow(x0 + 470, 140, 616, 140, ACCENT), box(620, 112, 190, 56, 'the probe with C = 1/8', fill=BOX, size=14),
          arrow(715, 170, 715, 196),
          text(600, 222, '3', 15, ACCENT, 'start', bold=True), text(620, 222, 'its test accuracy:', 14, INK, 'start'),
          text(620, 242, 'the score, measured once', 14, ACCENT, 'start', bold=True)]
    svg('probe_fit.svg', 830, 275, p)


def overview() -> None:
    names = [('texts', 'SST, IMDB', ''), ('tokenizer', 'text → token ids', 'tokenizer.md'),
             ('windows', 'packed, shuffled', 'data.md'), ('model', 'predicts next tokens', 'model.md'),
             ('loss + optimizer', 'nudge the weights', 'training.md'), ('checkpoint', 'saved files', 'checkpoints.md')]
    p = [text(10, 22, 'Training', 14, INK, 'start', bold=True)]
    for i, (name, sub, page) in enumerate(names):
        x = 10 + i * 160
        p += [box(x, 36, 136, 56, name, sub, fill=BOX_GREY if i == 0 else BOX)]
        p += [text(x + 68, 110, page, 11, ACCENT, mono=True)] if page else []
        p += [arrow(x + 138, 64, x + 156, 64)] if i < len(names) - 1 else []
    p += [f'<path d="M{10 + 4 * 160 + 40} 120 v14 H{10 + 3 * 160 + 68} v-14" fill="none" stroke="{GREY}" '
          f'stroke-width="2" marker-end="url(#g)"/>', text(10 + 3 * 160 + 148, 150, 'repeat for every batch', 12)]
    p += [text(10, 186, 'Using a trained model', 14, INK, 'start', bold=True)]
    p += [arrow(10 + 5 * 160 + 68, 120, 10 + 5 * 160 + 68, 200),
          box(10 + 5 * 160, 204, 136, 56, 'linear probe', 'the sentiment score'),
          text(10 + 5 * 160 + 68, 278, 'probe.md', 11, ACCENT, mono=True),
          f'<path d="M{10 + 5 * 160 + 68} 166 H{10 + 4 * 160 + 68} V200" fill="none" stroke="{GREY}" '
          f'stroke-width="2" marker-end="url(#g)"/>',
          box(10 + 4 * 160, 204, 136, 56, 'sampling', 'continue a prompt'),
          text(10 + 4 * 160 + 68, 278, 'sampling.md', 11, ACCENT, mono=True)]
    svg('overview.svg', 970, 292, p)


def tokenizer_bpe() -> None:
    x0 = 300
    rows = [['t', 'h', 'e'], ['␣', 't', 'h', 'e', 'n'], ['␣', 't', 'h', 'e']]

    def pieces(y, groups, hot=()):
        out, x = [], x0
        for g in groups:
            for t in g:
                w = 18 + 10 * len(t)
                out += [box(x, y, w, 30, t, fill=ACCENT_BOX if t in hot else BOX, size=14, mono=True)]
                x += w + 4
            x += 24
        return out
    p = step(42, '1', 'split the text into pieces', "GPT-2's regex: words, with their space")
    p += [text(x0, 26, '"the then the"', 14, INK, 'start', mono=True)]
    p += chips(x0, 34, ['the', '␣then', '␣the'], w=74, h=30, size=14)
    p += step(102, '2', 'each piece as bytes', 'the first 256 tokens: one per byte')
    p += pieces(88, rows)
    p += step(162, '3', 'count neighbouring pairs', 't h: 3 times, h e: 3 times, ␣ t: 2, e n: 1')
    p += [text(x0, 166, 'the most frequent: t h and h e tie, so the larger pair, t h, wins', 13, INK, 'start')]
    p += step(222, '4', 'merge it into a new token', 'token 256 = t + h')
    p += pieces(208, [['th', 'e'], ['␣', 'th', 'e', 'n'], ['␣', 'th', 'e']], hot=('th',))
    p += step(282, '5', 'repeat until vocab_size tokens', 'next: th + e = the (token 257)')
    p += pieces(268, [['the'], ['␣', 'the', 'n'], ['␣', 'the']], hot=('the',))
    svg('tokenizer_bpe.svg', 760, 314, p)


def tokenizer_vocab() -> None:
    p = [rect(20, 30, 220, 44, BOX_GREY), text(130, 57, '0 … 255: one per byte', 14, INK),
         rect(244, 30, 380, 44, BOX), text(434, 57, '256, 257, …: one per merge, in learned order', 14, INK),
         rect(628, 30, 92, 44, ACCENT_BOX), text(674, 57, '<bos>', 14, INK, mono=True),
         text(20, 20, 'token ids', 12, anchor='start'), text(720, 20, 'vocab_size tokens', 12, anchor='end'),
         text(20, 112, 'tokenizer.json', 13, INK, 'start', bold=True),
         text(20, 136, '{"merges": [[116, 104], [256, 101], ...]}', 14, INK, 'start', mono=True),
         text(20, 160, 'merge 0: token 116 (t) + token 104 (h) = token 256 (th);  '
              'merge 1: 256 (th) + 101 (e) = 257 (the)', 12, anchor='start')]
    svg('tokenizer_vocab.svg', 740, 176, p)


def data_packing() -> None:
    x0, w = 280, 66
    stream = ['<bos>', 'great', 'film', '<bos>', 'dull', '<bos>', 'loved', 'it', '<bos>']
    p = step(42, '1', 'the texts')
    p += chips(x0, 26, ['great film', 'dull', 'loved it'], w=110, size=14)
    p += step(102, '2', 'one stream, <bos> between texts', '<bos> also marks where a text ends')
    p += chips(x0, 86, stream, w=w - 4, grey=('<bos>',))
    p += step(172, '3', 'cut into windows', 'max_seq_len tokens each (here 4)')
    p += chips(x0, 146, stream[:4], w=w - 4, grey=('<bos>',))
    p += chips(x0 + 4 * w + 20, 146, stream[4:8], w=w - 4, grey=('<bos>',))
    p += chips(x0 + 8 * w + 40, 146, stream[8:], w=w - 4, fill=BOX_GREY)
    p += bracket(x0, x0 + 4 * w - 4, 180, 'window 1') + bracket(x0 + 4 * w + 20, x0 + 8 * w + 16, 180, 'window 2')
    p += [text(x0 + 8 * w + 70, 196, 'left over', 12)]
    p += step(262, '4', 'each token predicts the next one', 'the last has no next token in its window')
    p += chips(x0, 236, stream[:4], w=w - 4, grey=('<bos>',))
    targets = ['great', 'film', '<bos>', '—']
    for i, t in enumerate(targets):
        cx = x0 + i * w + 31
        p += [arrow(cx, 268, cx, 288), text(cx, 306, t, 13, ACCENT if t != '—' else GREY, mono=True)]
    p += [text(x0 + 4 * w + 10, 306, 'targets: its loss is dropped for the last', 12, anchor='start')]
    svg('data_packing.svg', 940, 322, p)


def data_mix() -> None:
    x0 = 300
    a = [f'A{i}' for i in range(1, 9)]
    p = step(42, '1', 'pack each dataset into windows', 'imdb: 8 windows, sst: 2 (weights 0.5 / 0.5)')
    p += chips(x0, 26, a, w=40) + chips(x0 + 8 * 44 + 24, 26, ['B1', 'B2'], w=40, fill=ACCENT_BOX)
    p += step(112, '2', "each gets its weight's share", 'an epoch has 10 windows: 5 + 5')
    p += chips(x0, 96, ['A2', 'A7', 'A4', 'A1', 'A5'], w=40)
    p += chips(x0 + 5 * 44 + 24, 96, ['B1', 'B2', 'B1', 'B2', 'B1'], w=40, fill=ACCENT_BOX)
    p += [text(x0 + 110, 146, 'a random 5 of 8', 12),
          text(x0 + 5 * 44 + 134, 146, 'whole passes, then a random part', 12)]
    p += step(192, '3', 'shuffle with seed + epoch', 'the same seed: the same order')
    order = ['B2', 'A4', 'B1', 'A2', 'B1', 'A7', 'A5', 'B2', 'A1', 'B1']
    p += [box(x0 + i * 44, 176, 40, 30, t, fill=ACCENT_BOX if t[0] == 'B' else BOX, size=13, mono=True)
          for i, t in enumerate(order)]
    p += step(262, '4', 'batch(step) reads the next ones', 'here batch_size = 4')
    p += [box(x0 + i * 44, 246, 40, 30, t, fill=ACCENT_BOX if t[0] == 'B' else BOX, size=13, mono=True)
          for i, t in enumerate(order)]
    p += bracket(x0, x0 + 4 * 44 - 4, 280, 'step 0') + bracket(x0 + 4 * 44, x0 + 8 * 44 - 4, 280, 'step 1')
    p += bracket(x0 + 8 * 44, x0 + 10 * 44 - 4, 280, 'step 2 …')
    p += [text(x0 + 10 * 44 + 10, 266, '… continues in epoch 2,', 12, anchor='start'),
          text(x0 + 10 * 44 + 10, 282, 'in a new order', 12, anchor='start')]
    svg('data_mix.svg', 920, 316, p)


def model_arch() -> None:
    line, bx = 420, 470

    def node(y):
        return [f'<circle cx="{line}" cy="{y}" r="11" fill="{BOX}" stroke="{GREY}" stroke-width="2"/>',
                text(line, y + 5, '+', 16, INK, bold=True)]
    p = step(42, '1', 'token ids', 'from the tokenizer')
    p += chips(line - 132, 26, ['<bos>', 'the', 'film', 'was'], w=62)
    p += [arrow(line, 58, line, 84)]
    p += step(106, '2', 'embed_tokens', 'a vector of hidden_dim numbers per token')
    p += [box(line - 132, 88, 264, 34, 'embed_tokens', fill=BOX_GREY, size=14), arrow(line, 124, line, 150)]
    p += step(180, '3', 'decoder layer', 'repeated num_layers times')
    p += [rect(line - 150, 150, 550, 200, 'none', stroke=GREY), text(line + 390, 168, '× num_layers', 13, anchor='end'),
          f'<path d="M{line} 150 V{350}" stroke="{GREY}" stroke-width="2"/>']
    for y, name, sub in ((186, 'RMSNorm → attention', 'each token looks at earlier tokens (RoPE, QK-norm)'),
                         (286, 'RMSNorm → MLP', 'each token alone (SwiGLU)')):
        p += [arrow(line, y, bx - 2, y), box(bx, y - 18, 310, 52, name, sub, size=14),
              f'<path d="M{bx + 140} {y + 34} V{y + 50} H{line + 13}" fill="none" stroke="{GREY}" stroke-width="2" '
              f'marker-end="url(#g)"/>', *node(y + 50)]
    p += [text(line - 16, 250, 'residual stream', 12, anchor='end'),
          text(line - 16, 266, '(the + keeps it)', 12, anchor='end')]
    p += [arrow(line, 350, line, 376)]
    p += step(398, '4', 'final norm')
    p += [box(line - 132, 380, 264, 34, 'norm (RMSNorm)', fill=BOX_GREY, size=14), arrow(line, 416, line, 442)]
    p += step(464, '5', 'lm_head', 'a score per token: what comes next?')
    p += [box(line - 132, 446, 264, 34, 'lm_head', size=14),
          text(line + 150, 468, 'tied: the same matrix as embed_tokens', 12, anchor='start')]
    svg('model.svg', 900, 496, p)


def training_loop() -> None:
    x0 = 330
    p = step(40, '1', 'logging, seeds, threads', 'resuming if log_dir has step checkpoints')
    p += [box(x0, 20, 420, 36, 'setup_logging, torch.manual_seed', fill=BOX_GREY, size=14)]
    p += step(100, '2', 'tokenizer, model, training data')
    p += [box(x0, 80, 420, 36, 'build_tokenizer, build_model, DatasetMix', fill=BOX_GREY, size=14)]
    p += step(160, '3', 'how long to train', 'num_epochs, or max_steps if sooner')
    p += [box(x0, 140, 420, 36, 'total_steps = num_steps(...)', fill=BOX_GREY, size=14)]
    p += step(220, '4', 'optimizers, lr schedule,', 'precision, held-out text')
    p += [box(x0, 200, 420, 36, 'optimizer.build, LambdaLR, autocast, eval_sets', fill=BOX_GREY, size=14)]
    p += [arrow(x0 + 210, 58, x0 + 210, 78), arrow(x0 + 210, 118, x0 + 210, 138), arrow(x0 + 210, 178, x0 + 210, 198),
          arrow(x0 + 210, 238, x0 + 258 - 48, 262)]
    p += step(300, '5', 'training loop', 'once per step')
    p += [rect(270, 266, 556, 140, 'none', stroke=GREY)]
    inner = [('batch', 46), ('loss', 40), ('backward', 64), ('clip', 36), ('optimizer step', 92), ('log, eval', 62),
             ('lr step', 52), ('save', 42)]
    xs = [282]
    for _, w in inner[:-1]:
        xs.append(xs[-1] + w + 14)
    for i, ((label, w), x) in enumerate(zip(inner, xs)):
        p += [box(x, 290, w, 34, label, size=11)]
        p += [arrow(x + w + 1, 307, xs[i + 1] - 1, 307)] if i < len(inner) - 1 else []
    p += [f'<path d="M{xs[-1] + 21} 326 V370 H{xs[0] + 23} V328" fill="none" stroke="{ACCENT}" stroke-width="2" '
          f'marker-end="url(#a)"/>', text(548, 390, 'next step, until total_steps', 12, ACCENT)]
    p += [arrow(x0 + 210, 408, x0 + 210, 430)]
    p += step(452, '6', 'save the final model')
    p += [box(x0, 432, 420, 36, 'model.safetensors', fill=BOX_GREY, size=14)]
    svg('training_loop.svg', 840, 482, p)


def lr_schedule() -> None:
    x0, y0, w, h, warm, floor = 150, 210, 480, 160, 0.1, 0.1
    X = lambda f: x0 + f * w
    Y = lambda v: y0 - v * h

    def curve(f):
        return min(1, (f + 1e-9) / warm) * (floor + (1 - floor) * (1 - f))

    def cos_curve(f):
        import math
        return min(1, (f + 1e-9) / warm) * (floor + (1 - floor) * (1 + math.cos(math.pi * f)) / 2)
    lin = ' '.join(f'{X(i / 200):.1f},{Y(curve(i / 200)):.1f}' for i in range(201))
    cos = ' '.join(f'{X(i / 200):.1f},{Y(cos_curve(i / 200)):.1f}' for i in range(201))
    p = [rect(X(0), Y(1), warm * w, h, BOX, rx=0),
         text(X(warm / 2), Y(1) - 8, 'warmup', 12),
         f'<path d="M{x0} {y0} H{x0 + w + 10} M{x0} {y0} V{y0 - h - 20}" stroke="{GREY}" stroke-width="2" '
         f'fill="none"/>',
         f'<polyline points="{cos}" fill="none" stroke="{GREY}" stroke-width="2" stroke-dasharray="6 4"/>',
         f'<polyline points="{lin}" fill="none" stroke="{ACCENT}" stroke-width="3"/>',
         f'<path d="M{x0} {Y(floor)} H{x0 + w}" stroke="{GREY}" stroke-width="1" stroke-dasharray="2 4"/>',
         text(x0 - 8, Y(1) + 5, 'lr', 13, INK, 'end'),
         text(x0 - 8, Y(floor) + 5, 'min_lr_ratio × lr', 12, anchor='end'),
         text(x0 - 8, y0 + 5, '0', 12, anchor='end'), text(x0 + w, y0 + 22, 'total_steps', 12, anchor='end'),
         text(x0, y0 + 22, 'step 0', 12, anchor='start'),
         text(X(0.55), Y(curve(0.55)) - 12, 'linear', 13, ACCENT, 'start', bold=True),
         text(X(0.62), Y(cos_curve(0.62)) + 26, 'cosine', 13, GREY, 'start')]
    svg('lr_schedule.svg', 680, 250, p)


def resume() -> None:
    x0 = 330
    p = step(42, '1', 'find the newest step checkpoint', 'latest_checkpoint(log_dir)')
    p += chips(x0, 26, ['step000200.pt', 'step000400.pt'], w=130, size=13)
    p += [rect(x0 + 134, 26, 130, 30, 'none', stroke=ACCENT)]
    p += step(102, '2', 'drop metrics after it', 'truncate_metrics: steps 401, 402, …')
    p += chips(x0, 86, ['… 300', '400', '450 ✕', '500 ✕'], w=90, size=13, grey=('450 ✕', '500 ✕'))
    p += step(162, '3', 'build everything as usual', 'tokenizer, model, optimizers, schedules')
    p += chips(x0, 146, ['tokenizer', 'model', 'optimizers', 'schedules'], w=90, size=13, fill=BOX_GREY)
    p += step(222, '4', 'load the saved state into them', 'load_train_state')
    p += [box(x0, 206, 184, 46, 'step000400.safetensors', 'the weights', size=13),
          box(x0 + 194, 206, 184, 46, 'step000400.pt', 'optimizers, schedules, rng', size=13)]
    p += step(290, '5', 'continue from step 401')
    p += [arrow(x0 + 92, 254, x0 + 92, 276), arrow(x0 + 286, 254, x0 + 286, 276),
          box(x0, 278, 378, 30, 'the training loop, as if it never stopped', size=13, fill=BOX_GREY)]
    svg('resume.svg', 740, 320, p)


def sampling() -> None:
    x0, words = 330, ['great', 'good', 'dull', 'the', 'bad', 'film']

    def bars(y, values, keep=None, hot=None):
        out = []
        for i, (word, v) in enumerate(zip(words, values)):
            colour = ACCENT if hot == i else BOX if keep is None else ACCENT_BOX if i < keep else BOX_GREY
            out += [rect(x0 + i * 62, y + 40 - v * 40, 46, v * 40, colour, rx=2),
                    text(x0 + 23 + i * 62, y + 56, word, 12, mono=True)]
        return out
    p = step(32, '1', 'the model scores every token', 'logits: the last position, one per token')
    p += bars(4, [0.9, 0.8, 0.55, 0.45, 0.35, 0.2])
    p += step(112, '2', 'softmax(scores / temperature)', 'probabilities; lower temperature: sharper')
    p += bars(84, [0.95, 0.7, 0.3, 0.2, 0.12, 0.05])
    p += step(192, '3', 'keep the top_k most likely', 'here top_k = 4')
    p += bars(164, [0.95, 0.7, 0.3, 0.2, 0.12, 0.05], keep=4)
    p += step(272, '4', 'keep the fewest covering top_p', 'here top_p = 0.9 of the probability')
    p += bars(244, [0.95, 0.7, 0.3, 0.2, 0.12, 0.05], keep=3)
    p += step(352, '5', 'draw one at random', 'seeded: SamplingParams.seed')
    p += bars(324, [0.95, 0.7, 0.3, 0.2, 0.12, 0.05], keep=3, hot=0)
    p += step(420, '6', 'append it, and repeat', 'until a stop token (<bos>) or max_tokens')
    svg('sampling.svg', 720, 440, p)


def experiments() -> None:
    p = [box(10, 30, 170, 56, 'tune_lr.py', 'build_runs(): 3 configs'), arrow(182, 58, 208, 58),
         box(212, 30, 210, 56, 'train_runs', 'named by what differs'), arrow(424, 58, 450, 58),
         rect(454, 22, 220, 72, BOX_GREY),
         text(564, 44, 'runs/tune_lr/', 13, INK, mono=True), text(564, 64, 'lr=0.01/  lr=0.003/', 12, mono=True),
         text(564, 82, 'lr=0.001/', 12, mono=True)]
    p += split(564, 96, [470, 690], 140)
    p += [box(380, 144, 180, 56, 'experiments.plot', 'plot.png or landscape.png'),
          box(600, 144, 180, 56, 'experiments.score', 'the probe: scores.png')]
    svg('experiments.svg', 800, 214, p)


def sae_how() -> None:
    # One token's activation through the SAE, top to bottom
    x0, features = 330, [0.05, 0.2, 0.9, 0.1, 0.0, 0.3, 0.05, 0.7, 0.15, 0.0, 0.1, 0.6, 0.05, 0.2, 0.0, 0.35, 0.1,
                         0.05, 0.8, 0.1, 0.0, 0.25, 0.1, 0.05, 0.15, 0.0, 0.4, 0.1, 0.05, 0.2]
    kept = sorted(range(len(features)), key=lambda i: -features[i])[:5]
    p = step(42, '1', "one token's activation", 'after one layer: hidden_dim numbers')
    p += vector(x0, 28, WAVE)
    p += [arrow(x0 + 54, 54, x0 + 54, 82), text(x0 + 66, 72, 'encoder', 12, anchor='start')]
    p += step(106, '2', 'a score for every feature', '2,048 features: many more than hidden_dim')
    p += vector(x0, 92, features, cell=12)
    p += [arrow(x0 + 54, 118, x0 + 54, 146), text(x0 + 66, 136, 'keep the k = 32 largest', 12, anchor='start')]
    p += step(170, '3', 'only a few features stay on', 'the rest set to 0: sparse')
    p += vector(x0, 156, [v if i in kept else 0.0 for i, v in enumerate(features)], cell=12)
    p += [arrow(x0 + 54, 182, x0 + 54, 210), text(x0 + 66, 200, 'decoder: add up their directions', 12,
                                                  anchor='start')]
    p += step(234, '4', 'the reconstruction', 'hidden_dim numbers again')
    p += vector(x0, 220, [min(1, v + 0.05) for v in WAVE])
    p += [rect(20, 270, 690, 40, BOX_GREY), text(365, 295, 'training: make 4 match 1, with only k features on '
                                                 '(no labels needed)', 14, INK)]
    svg('sae_how.svg', 730, 320, p)


def sae_steer() -> None:
    # Clamping a feature while the model writes, top to bottom
    x0 = 340
    p = step(42, '1', "the SAE layer's output, at every position", 'the model runs as usual up to here')
    p += vector(x0, 28, WAVE)
    p += split(x0 + 54, 52, [x0 + 75, x0 + 265], 82)
    p += step(102, '2', 'encode it, and keep the error', 'error = activation − reconstruction')
    p += [box(x0, 84, 150, 36, 'features', size=14), box(x0 + 170, 84, 190, 36, 'error (what the SAE misses)',
                                                          fill=BOX_GREY, size=13)]
    p += step(162, '3', 'set one feature', 'e.g. #904 to 1 × its largest activation')
    p += [box(x0, 144, 150, 36, '#904 = 1 × max', size=14), arrow(x0 + 75, 122, x0 + 75, 142)]
    p += step(222, '4', 'decode, and add the error back', 'only the clamped feature changes')
    p += [box(x0, 204, 360, 36, 'decoder(features) + error', size=14), arrow(x0 + 75, 182, x0 + 75, 202),
          arrow(x0 + 265, 122, x0 + 265, 202)]
    p += step(282, '5', 'the rest of the model continues', 'the next token now leans positive')
    p += [arrow(x0 + 180, 242, x0 + 180, 262), box(x0, 264, 360, 36, 'the later layers → next token → repeat', size=14,
                                                   fill=BOX_GREY)]
    svg('sae_steer.svg', 720, 312, p)


if __name__ == '__main__':
    for draw in (pipeline, features, fit, overview, tokenizer_bpe, tokenizer_vocab, data_packing, data_mix,
                 model_arch, training_loop, lr_schedule, resume, sampling, experiments, sae_how, sae_steer):
        draw()
