# Staff model

The organisers' trained model for the main task, after extensive tuning: **81.5% on the SST test set**. Use it to
see how far the task can go; teams are ranked against each other.

This folder has only the saved model, not the training code:

| File | What it is |
|---|---|
| `model.safetensors` | A microgpt checkpoint: the torch model (`microgpt/pt/model.py`), pretrained without labels |
| `tokenizer.json` | Its byte-level BPE tokenizer |
| `model.probe.joblib` | The linear probe, fitted on the SST training set |

```
python3 -m scripts.sample_lm checkpoint=staff_model/model.safetensors       # type a review: its sentiment and how the model continues it
python3 -m scripts.linear_probe model=staff_model/model.safetensors         # check the staff score yourself
```
