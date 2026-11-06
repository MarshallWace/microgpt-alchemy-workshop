# The linear probe: how much sentiment has a model learned from unlabelled text? Freeze the model, turn each text into a
# feature vector (its last layer at the last token), and fit only an L1-regularised logistic regression on the labels
# (SST-2 by default), as OpenAI's sentiment neuron did. The regularisation strength is picked on dev; the score is test
# accuracy. scripts/linear_probe.py runs it from the command line.
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from microgpt.checkpoint import load_checkpoint
from microgpt.config import BACKENDS
from microgpt.load_data import load_split
from microgpt.tokenizer import checkpoint_tokenizer

Featurize = Callable[[list[str]], np.ndarray]  # texts -> features [n_texts, n_features]
L1_STRENGTHS = tuple(2.0 ** np.arange(-6, 3))  # scikit-learn's C: lower means stronger L1 regularisation


def fit_probe(Xtr, ytr, Xdv, ydv) -> tuple[float, Pipeline]:
    # Standardised features, then L1-regularised logistic regression with the strength picked on dev
    probes = [make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=1, C=c, solver='liblinear')).fit(Xtr, ytr)
              for c in L1_STRENGTHS]
    return max(((probe.score(Xdv, ydv), probe) for probe in probes), key=lambda pair: pair[0])


def load_features(featurize: Featurize, dataset: str) -> tuple[dict, dict]:
    splits = {s: load_split(dataset, s) for s in ('train', 'dev', 'test')}
    X = {s: np.asarray(featurize([t for t, _ in d]), dtype=np.float64) for s, d in splits.items()}
    return X, {s: np.array([label for _, label in d]) for s, d in splits.items()}


def best_single_feature(X: dict, y: dict) -> tuple[int, dict[str, float]]:
    # A "sentiment neuron"? Each feature alone, thresholded halfway between the class means
    pos, neg = X['train'][y['train'] == 1].mean(0), X['train'][y['train'] == 0].mean(0)
    acc = {s: (((X[s] - (pos + neg) / 2) * np.sign(pos - neg) > 0) == y[s][:, None]).mean(0) for s in ('dev', 'test')}
    unit = int(np.argmax(acc['dev']))
    return unit, {s: float(a[unit]) for s, a in acc.items()}


def data_efficiency(X: dict, y: dict, sizes=(20, 100, 500, 2000), repeats: int = 5) -> dict[int, tuple[float, float]]:
    # Test accuracy (mean, std over random subsets) with only n labelled training examples
    rng, out = np.random.RandomState(0), {}
    for n in sizes:
        accs = []
        for _ in range(repeats):
            sub = rng.choice(len(y['train']), n, replace=False)
            _, probe = fit_probe(X['train'][sub], y['train'][sub], X['dev'], y['dev'])
            accs.append(probe.score(X['test'], y['test']))
        out[n] = (float(np.mean(accs)), float(np.std(accs)))
    return out


def evaluate_probe(featurize: Featurize, dataset: str = 'sst') -> dict:
    X, y = load_features(featurize, dataset)
    print(f"{X['train'].shape[1]} features, {len(y['train'])} training labels")
    dev_acc, probe = fit_probe(X['train'], y['train'], X['dev'], y['dev'])
    test_acc, classifier = probe.score(X['test'], y['test']), probe[-1]
    print(f"linear probe: dev {dev_acc:.1%}, test {test_acc:.1%} "
          f"(C={classifier.C:g}, {np.count_nonzero(classifier.coef_)} weights used)")
    unit, unit_acc = best_single_feature(X, y)
    print(f"best single feature: #{unit}, dev {unit_acc['dev']:.1%}, test {unit_acc['test']:.1%}")
    for n, (acc, spread) in data_efficiency(X, y).items():
        print(f"  {n:>5} labels: test {acc:.1%} ± {spread:.1%}")
    # probe.predict_proba(featurize(texts))[:, 1]: each text's probability of being positive
    return {'dev_acc': float(dev_acc), 'test_acc': float(test_acc), 'unit': unit, 'unit_test_acc': unit_acc['test'],
            'probe': probe}


def featurizer(model, tokenizer) -> Featurize:
    # The model's last layer at the last token, after reading <bos> and the text's last max_seq_len - 1 tokens (all of
    # nearly every SST sentence)
    bos, n = tokenizer.bos_token_id, model.config.max_seq_len - 1

    def featurize(texts: list[str]) -> np.ndarray:
        return np.array([model.hidden_states([bos] + tokenizer.encode(text)[-n:])[-1] for text in texts])
    return featurize


def checkpoint_featurizer(checkpoint: str) -> Featurize:
    # The checkpoint's model, with the tokenizer.json in its folder
    for backend in BACKENDS.values():
        backend.set_threads(1)  # one text at a time: matrices this small are fastest on one thread
    return featurizer(load_checkpoint(checkpoint), checkpoint_tokenizer(checkpoint))


def score(checkpoint: str) -> dict[str, float | None]:
    # The probe's dev and test accuracy for a checkpoint; None for a run that diverged (its features are not finite)
    try:
        result = evaluate_probe(checkpoint_featurizer(checkpoint))
    except ValueError:
        return {'dev_acc': None, 'test_acc': None}
    return {'dev_acc': result['dev_acc'], 'test_acc': result['test_acc']}


def sentiment(checkpoint: str, model, tokenizer):
    # Returns text -> the probe's probability that it is positive
    featurize, path = featurizer(model, tokenizer), Path(checkpoint).with_suffix('.probe.joblib')
    if not path.exists():
        print('fitting the probe on SST, once...')
        joblib.dump(evaluate_probe(featurize)['probe'], path)
    probe = joblib.load(path)
    return lambda text: float(probe.predict_proba(featurize([text]))[0, 1])
