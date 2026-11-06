# Sample from a trained model: how it continues a prompt, and how positive the linear probe thinks the prompt is (the
# probe is fitted on SST the first time, about a minute, and saved next to the checkpoint)
#   python3 -m scripts.sample_lm checkpoint=runs/pretrain/model.safetensors                          # type prompts
#   python3 -m scripts.sample_lm checkpoint=runs/pretrain/model.safetensors prompt='the acting was'  # or give one
#   python3 -m scripts.sample_lm checkpoint=runs/pretrain/model.safetensors probe=false sampling_params.temperature=0.5
import chz

from alchemy.checkpoint import load_checkpoint
from alchemy.probe import sentiment
from alchemy.sampling import SamplingParams, sample
from alchemy.tokenizer import checkpoint_tokenizer


@chz.chz
class SampleLMConfig:
    checkpoint: str
    prompt: str = ''           # without one, type prompts
    num_samples: int = 1
    sampling_params: SamplingParams = chz.field(default_factory=lambda: SamplingParams(max_tokens=30, temperature=0.8))
    probe: bool = True         # also print the probe's sentiment for the prompt


def main(config: SampleLMConfig) -> None:
    model = load_checkpoint(config.checkpoint)
    tokenizer = checkpoint_tokenizer(config.checkpoint)
    positive = sentiment(config.checkpoint, model, tokenizer) if config.probe else None
    for prompt in [config.prompt] if config.prompt else iter(lambda: input('prompt> '), ''):
        prompt_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
        response = sample(model, prompt_ids, config.num_samples, config.sampling_params, tokenizer.eos_token_id)
        if positive:
            print(f'positive: {positive(prompt):.0%}')
        for sequence in response.sequences:
            print(f'{prompt} ... {tokenizer.decode(sequence.tokens)}')


if __name__ == '__main__':
    main(chz.entrypoint(SampleLMConfig))
