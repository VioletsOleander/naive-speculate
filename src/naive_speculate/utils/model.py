from typing import cast

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM, TextStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast


class QwenModel:
    """Wrapper class for Qwen3ForCausalLM to simplify inference API and typing annotations."""

    model: Qwen3ForCausalLM
    model_config: Qwen3Config
    kv_cache: DynamicCache

    def __init__(self, model_name: str):
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )

        self.model_config = self.model.config

        self.kv_cache = DynamicCache(config=self.model.config)

    def inference(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        streamer: TextStreamer | None = None,
    ):
        input_ids = self.prefill(
            input_ids,
            max_new_tokens,
            decode_method,
            streamer,
        )
        outputs = self.decode(
            input_ids,
            max_new_tokens - 1,  # prefill already generated 1 token
            decode_method,
            streamer,
        )
        return outputs

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return input_ids

        # 1. Forward
        # kv_cache is updated inside model.forward as a side effect
        outputs = self.model.forward(
            input_ids=cast(torch.LongTensor, input_ids),
            logits_to_keep=0,  # output all logits
            use_cache=True,
            past_key_values=self.kv_cache,
        )

        # 2. Sample
        next_token_ids = self._sample(outputs, decode_method)

        # 3. Update
        input_ids = torch.cat([input_ids, next_token_ids], dim=-1)
        if streamer is not None:
            streamer.put(next_token_ids.cpu())

        return input_ids

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            if streamer is not None:
                streamer.end()
            return input_ids

        num_new_tokens = 0
        while True:
            # 1. Forward
            # kv_cache is updated inside model.forward as a side effect
            outputs = self.model.forward(
                input_ids=cast(torch.LongTensor, input_ids[:, -1:]),
                logits_to_keep=1,
                use_cache=True,
                past_key_values=self.kv_cache,
            )

            # 2. Sample
            next_token_ids = self._sample(outputs, decode_method)

            # 3. Update
            input_ids = torch.cat([input_ids, next_token_ids], dim=-1)
            num_new_tokens += 1
            if streamer is not None:
                streamer.put(next_token_ids.cpu())

            if (
                num_new_tokens >= max_new_tokens
                or next_token_ids[0, 0].item() == self.model_config.eos_token_id
            ):
                break

        if streamer is not None:
            streamer.end()
        return input_ids

    def _sample(
        self, outputs: CausalLMOutputWithPast, decode_method: str
    ) -> torch.Tensor:
        assert outputs.logits is not None

        # trick adopted from transformers.GenerationMixin._sample:
        # copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)

        match decode_method:
            case "greedy":
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            case "random":
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)
            case _:
                raise ValueError(f"Unknown decode method: {decode_method}")

        return next_token_ids

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __str__(self) -> str:
        return str(self.model)
