# Sample with SAE features clamped, next to the same samples without: try what a feature does
#   python3 -m experiments.sae.steer checkpoint=runs/pretrain/model.safetensors clamp.904=1 prompt='this movie'
#   python3 -m experiments.sae.steer checkpoint=... clamp.904=0 clamp.1067=1.5       # several; 0 switches one off
#   python3 -m experiments.sae.steer checkpoint=... clamp.904=1                      # no prompt: type prompts
# experiments.sae.serve does the same from the web page.
import chz

from alchemy.sampling import SamplingParams
from experiments.sae.sae import TopKSAE, load_run, steer


@chz.chz
class SteerConfig:
    checkpoint: str                # the model the SAE was trained on
    sae: str = 'runs/sae/sae.safetensors'
    clamp: dict[str, float] = chz.field(default_factory=dict)  # feature -> times its max activation, e.g. clamp.904=1
    prompt: str = ''               # without one, type prompts
    num_samples: int = 2
    sampling_params: SamplingParams = chz.field(default_factory=lambda: SamplingParams(max_tokens=40, temperature=0.8))


def main(config: SteerConfig) -> None:
    model, tokenizer = load_run(config.checkpoint)
    sae = TopKSAE.load(config.sae)
    clamps = {int(feature): times for feature, times in config.clamp.items()}
    for prompt in [config.prompt] if config.prompt else iter(lambda: input('prompt> '), ''):
        result = steer(model, tokenizer, sae, prompt, clamps, config.num_samples, config.sampling_params)
        for label in ('clamped', 'normal'):
            for text in result[label]:
                print(f'[{label}] {prompt}{text}')


if __name__ == '__main__':
    main(chz.entrypoint(SteerConfig))
