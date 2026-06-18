from dataclasses import dataclass
from typing import Optional

from evalplus.provider.base import DecoderBase


@dataclass
class ModelConfig:
    model: str
    backend: str
    dataset: str
    batch_size: int = 1
    temperature: float = 0.0
    force_base_prompt: bool = False
    instruction_prefix: Optional[str] = None
    response_prefix: Optional[str] = None
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    tp: int = 1
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    base_url: Optional[str] = None
    attn_implementation: str = "eager"
    device_map: Optional[str] = None
    gptqmodel_backend: str = "auto"
    gguf_file: Optional[str] = None


def make_model_from_config(config: ModelConfig):
    return make_model(
        model=config.model,
        backend=config.backend,
        dataset=config.dataset,
        batch_size=config.batch_size,
        temperature=config.temperature,
        force_base_prompt=config.force_base_prompt,
        instruction_prefix=config.instruction_prefix,
        response_prefix=config.response_prefix,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        tp=config.tp,
        enable_prefix_caching=config.enable_prefix_caching,
        enable_chunked_prefill=config.enable_chunked_prefill,
        base_url=config.base_url,
        attn_implementation=config.attn_implementation,
        device_map=config.device_map,
    )


def make_model(
        model: str,
        backend: str,
        dataset: str,
        batch_size: int = 1,
        temperature: float = 0.0,
        force_base_prompt: bool = False,
        # instruction model only
        instruction_prefix=None,
        response_prefix=None,
        # non-server only
        dtype="bfloat16",
        trust_remote_code=False,
        # vllm only
        tp=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        # openai only
        base_url=None,
        # hf only
        attn_implementation="eager",
        device_map=None,
        # gptqmodel only
        gptqmodel_backend: str = 'auto',
        gguf_file: str = None,
) -> DecoderBase:
    if backend == "openai":
        from evalplus.provider.openai import OpenAIChatDecoder

        assert not force_base_prompt, f"{backend} backend does not serve base model"
        return OpenAIChatDecoder(
            name=model,
            batch_size=batch_size,
            temperature=temperature,
            base_url=base_url,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
        )
