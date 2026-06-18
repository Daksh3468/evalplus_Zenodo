import os
from typing import List, Optional

import openai

from evalplus.gen.util import openai_request
from evalplus.provider.base import DecoderExtra
from evalplus.provider.utility import concurrent_call


class OpenAIChatDecoder(DecoderExtra):
    def __init__(self, name: str, base_url=None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self._usage_of_last_request = None

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200, extra_body=None) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        assert num_samples > 0, "Number of samples must be positive"
        batch_size = min(self.batch_size, num_samples)

        if self.instruction_prefix is not None:  # If prefix is none, it means we only want to use the prompt
            prompt = self.instruction_prefix + f"\n```python\n{prompt.strip()}\n```"

        if "deespeek" in self.name:
            self.max_new_tokens = 15000
        elif "starcoder" in self.name:
            self.max_new_tokens = 5000
        # use concurrency based batching for o1 and deepseek models
        # if self.name.startswith("o1-") or self.name == "deepseek-chat":
        #     return self._codegen_batch_via_concurrency(prompt, num_samples, extra_body)

        return self._codegen_api_batch(prompt, batch_size, extra_body=extra_body)

    def get_usage_of_last_request(self) -> Optional[dict]:
        return self._usage_of_last_request

    def _codegen_api_batch(self, prompt: str, batch_size: int, extra_body) -> List[str]:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=self.base_url
        )

        ret = openai_request.make_auto_request(
            client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=batch_size,
            extra_body=extra_body
        )

        self._usage_of_last_request = ret.usage.to_dict()

        outputs = []
        for item in ret.choices:
            outputs.append(item.message.content)

        return outputs

    def _codegen_batch_via_concurrency(self, prompt: str, batch_size: int, extra_body) -> List[str]:
        batches = concurrent_call(
            batch_size, self._codegen_api_batch, prompt, batch_size=1, extra_body=extra_body
        )
        return [b[0] for b in batches]

    def is_direct_completion(self) -> bool:
        return False
