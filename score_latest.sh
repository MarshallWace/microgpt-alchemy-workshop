#!/bin/zsh
# Score the newest checkpoint of the long run with our featurizer: ./score_latest.sh
cd "$(dirname "$0")"
CKPT=$(ls runs/*/step*.safetensors 2>/dev/null | tail -1)
[ -f runs/final/model.safetensors ] && CKPT=runs/final/model.safetensors
[ -z "$CKPT" ] && { echo "no checkpoint yet (first one at step 2000, ~10 min in)"; exit 1; }
echo "scoring $CKPT"
CHECKPOINT=$CKPT .venv/bin/python -m scripts.linear_probe model=my_model.featurize 2>&1 | grep -E "features|linear probe|best single"
