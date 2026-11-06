# Serves the SAE page (runs/sae/index.html, from experiments.sae.article) with a live model behind it, so the steering
# explorer can take your own prompts and clamp any features. Standard library only.
#   python3 -m experiments.sae.serve checkpoint=runs/pretrain/model.safetensors        # then open http://localhost:8000
# POST /api/steer {"prompt": "...", "clamps": {"904": 1.0}, "num_samples": 2, "max_tokens": 40, "temperature": 0.8,
# "seed": 0} -> {"clamped": [...], "normal": [...]}, the same seeds with and without the clamps.
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chz
import torch

from alchemy.sampling import SamplingParams
from experiments.sae.sae import TopKSAE, load_run, steer


@chz.chz
class ServeConfig:
    checkpoint: str              # the model the SAE was trained on
    sae: str = 'runs/sae/sae.safetensors'
    page: str = 'runs/sae/index.html'
    host: str = '0.0.0.0'
    port: int = 8000


def parse(body: dict, num_features: int) -> tuple[str, dict[int, float], int, SamplingParams]:
    # A steering request, with limits so one request cannot tie the model up for long
    prompt = str(body.get('prompt', ''))[:500]
    clamps = {int(f): max(-5.0, min(5.0, float(s))) for f, s in dict(body.get('clamps', {})).items()}
    if len(clamps) > 5 or any(not 0 <= f < num_features for f in clamps):
        raise ValueError(f'up to 5 clamps, of features 0 to {num_features - 1}')
    params = SamplingParams(max_tokens=max(1, min(80, int(body.get('max_tokens', 40)))),
                            temperature=max(0.0, min(2.0, float(body.get('temperature', 0.8)))),
                            seed=int(body.get('seed', 0)))
    return prompt, clamps, max(1, min(4, int(body.get('num_samples', 2)))), params


def main(config: ServeConfig) -> None:
    torch.set_num_threads(4)
    model, tokenizer = load_run(config.checkpoint)
    sae = TopKSAE.load(config.sae)
    page, lock = Path(config.page).read_bytes(), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ('/', '/index.html'):
                self.reply(200, page, 'text/html; charset=utf-8')
            elif self.path == '/api/health':
                self.reply(200, json.dumps({'features': len(sae.max_act)}).encode(), 'application/json')
            else:
                self.reply(404, b'not found', 'text/plain')

        def do_POST(self) -> None:
            if self.path != '/api/steer':
                return self.reply(404, b'not found', 'text/plain')
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get('Content-Length', 0)), 10_000)))
                prompt, clamps, num_samples, params = parse(body, len(sae.max_act))
            except (ValueError, TypeError, AttributeError) as error:
                return self.reply(400, json.dumps({'error': str(error)}).encode(), 'application/json')
            with lock:  # one generation at a time
                result = steer(model, tokenizer, sae, prompt, clamps, num_samples, params)
            self.reply(200, json.dumps(result).encode(), 'application/json')

    print(f'serving {config.page} on http://{config.host}:{config.port}')
    ThreadingHTTPServer((config.host, config.port), Handler).serve_forever()


if __name__ == '__main__':
    main(chz.entrypoint(ServeConfig))
