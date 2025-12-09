from typing import cast

import torch
from transformers import DynamicCache, Qwen3ForCausalLM, TextStreamer


class QwenModel:
    """
    Wrapper class for Qwen3ForCausalLM that provides convenient methods for model loading,
    inference with key-value (KV) caching, and support for different decoding methods.

    This class handles:
        - Loading a Qwen3ForCausalLM model from a given model name.
        - Managing the model's KV cache for efficient autoregressive inference.
        - Performing inference using either greedy or random decoding.
        - Streaming output tokens via an optional TextStreamer.

    Key parameters:
        - model_name (str): Name or path of the pretrained Qwen model to load.
        - decode_method (str): Decoding strategy, either "greedy" or "random".
        - max_new_tokens (int): Maximum number of tokens to generate.
        - streamer (TextStreamer, optional): If provided, streams generated tokens.

    Example usage:
        >>> model = QwenModel("Qwen/Qwen1.5-0.5B")
        >>> input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids
        >>> output_ids = model.inference(input_ids, max_new_tokens=20, decode_method="greedy")

    Attributes:
        model (Qwen3ForCausalLM): The underlying Qwen model.
        kv_cache (DynamicCache): KV cache for efficient decoding.
        eos_token_id (int): End-of-sequence token id.
    """
    def __init__(self, model_name: str):
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )

        assert self.model.config.eos_token_id is not None
        self.eos_token_id = self.model.config.eos_token_id

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

    def _sample(self, logits: torch.Tensor, decode_method: str) -> torch.Tensor:
        match decode_method:
            case "greedy":
                next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
            case "random":
                probs = torch.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)
            case _:
                raise ValueError(f"Unknown decode method: {decode_method}")
        return next_token_ids

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        streamer: TextStreamer | None = None,
    ):
        num_new_tokens = 0
        while True:
            if num_new_tokens >= max_new_tokens:
                break

            # 1. Forward
            # kv_cache is updated inside model.forward as a side effect
            outputs = self.model.forward(
                input_ids=cast(torch.LongTensor, input_ids[:, -1:]),
                attention_mask=torch.ones_like(input_ids[:, -1:]),
                logits_to_keep=1,
                use_cache=True,
                past_key_values=self.kv_cache,
            )

            # 2. Sample
            # trick adopted from transformers.GenerationMixin._sample: copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            assert outputs.logits is not None
            next_token_logits = outputs.logits[:, -1, :].to(
                copy=True, dtype=torch.float32
            )
            print(f"Iteration {num_new_tokens}: next_token_logits={next_token_logits}")
            next_token_ids = self._sample(next_token_logits, decode_method)

            if streamer is not None:
                streamer.put(next_token_ids.cpu())

            # 3. Update
            input_ids = torch.cat([input_ids, next_token_ids], dim=-1)
            num_new_tokens += 1

            if next_token_ids[0, 0].item() == self.eos_token_id:
                break

        if streamer is not None:
            streamer.end()

        return input_ids

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        streamer: TextStreamer | None = None,
    ):
        if max_new_tokens <= 0:
            return input_ids

        # 1. Forward
        # kv_cache is updated inside model.forward as a side effect
        outputs = self.model.forward(
            input_ids=cast(torch.LongTensor, input_ids),
            attention_mask=torch.ones_like(input_ids),
            logits_to_keep=0,  # output all logits
            use_cache=True,
            past_key_values=self.kv_cache,
        )

        # 2. Sample
        # trick adopted from transformers.GenerationMixin._sample: copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        assert outputs.logits is not None
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)
        next_token_ids = self._sample(next_token_logits, decode_method)
        input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

        if streamer is not None:
            streamer.put(next_token_ids.cpu())

        return input_ids

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __str__(self) -> str:
        return str(self.model)
