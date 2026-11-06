# 9. Sparse autoencoders

*The probe showed that sentiment can be read from the model's **activations** (the vectors it computes inside, as it
reads), but only because we asked about it, with labels. What does the model represent on its own? A sparse
autoencoder lists the features it uses, without labels, and lets us turn one up to see what it does.*

## Why: probes answer our questions, SAEs show the model's

| | linear probe (chapter 7) | sparse autoencoder (SAE) |
|---|---|---|
| asks | "is *sentiment* readable here?" | "what directions does the model use here?" |
| needs labels | yes (SST) | no: only activations |
| finds | one direction, for one question | thousands of **features**, each ideally meaning one thing |

The obvious place to look, single numbers ("neurons") inside the model, usually mixes several concepts: there are
more things to represent than numbers to hold them. Anthropic's
[Towards](https://transformer-circuits.pub/2023/monosemantic-features) and
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity) showed that rewriting
activations as a **sparse** combination of many directions gives features that are much easier to read, and that
clamping one steers the model (Golden Gate Claude, which turned up a "Golden Gate Bridge" feature).

## How: rebuild each activation from a few features

![four steps from top to bottom: 1 one token's activation at the SAE's layer, hidden_dim numbers; 2 the encoder gives a score for each of 2,048 features; 3 only the 32 largest are kept, the rest set to 0; 4 the decoder adds up those features' directions into a reconstruction of hidden_dim numbers; training makes the reconstruction match the activation](figures/sae_how.svg)

- An **activation** here is the residual stream after one layer (`layer`, by default the last) at one token: the
  `hidden_dim` numbers the model passes from layer to layer.
- [`TopKSAE`](../experiments/sae/sae.py#L33-L65) has 2,048 features. [`encode`](../experiments/sae/sae.py#L46-L49)
  scores them all and keeps the **k = 32 largest** (a TopK SAE, as in
  [Gao et al. 2024](https://arxiv.org/abs/2406.04093)); [`decode`](../experiments/sae/sae.py#L51-L52) adds up
  those features' directions.
- It is trained only to make the reconstruction match the activation, on 1.5 million IMDB tokens
  ([`main`](../experiments/sae/train.py#L36-L75)): no labels. Keeping only 32 features on forces each to be useful,
  which is what makes them readable.

## Does it work?

Three checks on held-out text, printed by `train.py` (numbers for the example model the rest of this page uses):

| check | means | example model |
|---|---|---|
| variance explained | how much of the activations the reconstruction keeps | 87.0% |
| dead features | features that never fire: wasted | 0% |
| loss recovered | the model's next-token loss with the SAE's layer replaced by the reconstruction, compared with replacing it by its average ([`lm_loss`](../experiments/sae/train.py#L78-L89)): 100% = no harm | 96.7% |

So the 32-of-2,048 rewrite keeps almost everything the rest of the model needs.

## Reading features: one page

[`article.py`](../experiments/sae/article.py#L82-L162) writes `runs/sae/index.html`, one page laid out like
Transformer Circuits' *Towards Monosemanticity* (template
[`article_template.html`](../experiments/sae/article_template.html), data inside the page, so it opens offline). It
has three sections:

| section | what it does |
|---|---|
| **article** | the results written up: this SAE's quality, three features investigated in detail (which reviews they fire in, examples at each strength, what they push the next token towards, steering), features against raw dimensions, and caveats; numbers and figures come from the data |
| **steering explorer** | pick a feature, a strength and a prompt to see samples clamped next to normal; opened as a file it shows precomputed samples for 20 features, served it takes your own prompt and any features (below) |
| **feature browser** | all 2,048 features, each with an automatic **label**: the words most often strongly shaded in its strongest examples, punctuation skipped ([`browser`](../experiments/sae/article.py#L68-L72)); search by number or word (`904`, `funny`), sort by SST accuracy or how often it fires |

Clicking a feature opens its view:

| in a feature view | what it shows |
|---|---|
| fires on, largest activation | how often it is among a token's 32, and how strongly it can fire |
| alone on SST | how well it alone tells positive from negative ([`single_accuracy`](../experiments/sae/sae.py#L120-L125)); 50% = chance |
| histogram | how strongly it fires when it does, relative to its largest activation |
| pushes towards / away from | the next tokens its direction makes more or less likely, through the final norm's scale and `lm_head` ([`effects`](../experiments/sae/article.py#L92)): approximate, as it skips the layers after the SAE's |
| strongest examples | text where it fires most, each token shaded by its activation ([`context`](../experiments/sae/sae.py#L99-L105)); the token it fires most on is underlined; hover for the value |
| about half as strong | text where it fires below half its largest: does the meaning hold? |
| steer this feature | fills in the live steering form |

Some features of the example model, with their automatic labels. Feature numbers are specific to one model and one
SAE: on yours, the same kinds of feature appear under other numbers.

| feature | label | what it is | fires on | alone on SST | pushes towards |
|---|---|---|---|---|---|
| **#904** | and · very · warm | warm praise: "lovely, warm, sentimental", "a touching and" | 18.5% of tokens | 73.9% (positive) | heartwarming, enchanting, wonderful |
| **#1241** | solid · plausible · excellent | approving adjectives | 1.9% | 67.9% (positive) | cast, ensemble, performances |
| **#1067** | poorly · dull · uninteresting | complaints: "incoherent", "amateurish" | 2.3% | 59.6% (negative) | caricatures, uninteresting, characters |

One feature alone is never as good as the probe: sentiment is spread over many features.

## Steering: clamp a feature

![five steps from top to bottom: 1 the SAE layer's output; 2 the SAE encodes it into features, and the error it misses is kept; 3 one feature is set, e.g. #904 to 1 times its largest activation; 4 decode and add the error back; 5 the rest of the model continues to the next token](figures/sae_steer.svg)

[`clamp_features`](../experiments/sae/sae.py#L68-L79) does this at every position while the model writes. Adding
the error back means only the clamped feature changes. For "i thought the acting was":

| clamp | continues |
|---|---|
| none | "a bit over the top, but really it was bad…" |
| `clamp.904=1` (warm praise) | "superb. the film has life and good passion… great performances and the great acting" |
| `clamp.1067=1` (complaints) | "passable, and the film over was predictable. the dialog was so flat and completely unbelievable" |

About **1×** a feature's largest activation steers while staying fluent; 2× and more breaks the grammar into lists
of sentiment words.

## Run it

From the torch-alchemy folder, on a trained model:

```
python3 -m experiments.sae.train checkpoint=runs/pretrain/model.safetensors             # ~1 min: saves runs/sae/sae.safetensors
python3 -m experiments.sae.article checkpoint=runs/pretrain/model.safetensors        # ~4 min: runs/sae/index.html
python3 -m experiments.sae.serve checkpoint=runs/pretrain/model.safetensors          # then open http://localhost:8000
```

Opened as a file, `runs/sae/index.html` steers only with precomputed samples. Served by
[`serve.py`](../experiments/sae/serve.py#L39-L78), the same page gets a box for your own prompt and clamps for any
features: the page sends them to the running model (`POST /api/steer`), which answers one request at a time, with at
most 4 samples, 80 tokens and 5 clamps.

The same steering from the terminal:

```
python3 -m experiments.sae.steer checkpoint=... clamp.904=1 prompt='i thought the acting was'
python3 -m experiments.sae.steer checkpoint=... clamp.904=0 clamp.1067=1                 # several; 0 switches one off
```

[`steer.py`](../experiments/sae/steer.py#L22-L30), the server and the page's precomputed samples all use one
[`steer`](../experiments/sae/sae.py#L143-L147): samples clamped and not, with the same seeds, so they differ only by
the clamp; without `prompt` it asks for prompts.

**Compared with Transformer Circuits:** their articles and feature browsers are static pages; the open equivalents
are SAEDashboard pages and [Neuronpedia](https://www.neuronpedia.org), which also steers open models live, as
`serve.py` does here.

That is the whole pipeline: text to tokens, tokens to batches, a model that learns to predict the next token,
saving it, letting it write, measuring what it learned, finding better settings, and looking inside it.

---

← Previous: [8. Experiments](08-experiments.md) · Next: [back to the start](README.md) →
