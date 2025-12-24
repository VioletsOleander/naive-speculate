# TODO: Refactor inference basis
from typing import cast

import torch
from transformers import (
    DynamicCache,
    DynamicLayer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from naive_speculate.utils import DecodeMethod

from ._output import DecodeOutput, OutputCollection, PrefillOutput


class QwenModel:
    """Wrapper class for Qwen3ForCausalLM to simplify inference API and typing annotations.

    Currently only supports `batch_size=1`. Maybe later on I can add batch support, and further
    considering intergration with continuous batching, pd disaggregation, etc.

    Attributes:
        model (Qwen3ForCausalLM): `transformers` model instance.
        model_config (Qwen3Config): `transformers` model config instance.
        kv_cache (DynamicCache): `transformers` KV cache instance.
        decode_chunk_size (int): EOS token check interval during decoding, default to 8.
            Used as a simple trick to reduce device synchronization overhead.
    """

    model: Qwen3ForCausalLM
    model_config: Qwen3Config
    kv_cache: DynamicCache
    decode_chunk_size: int

    def __init__(self, model_name: str) -> None:
        """Initialize the QwenModel.

        Args:
            model_name (str): Used to load the pre-trained model.
        """
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )
        self.model_config = self.model.config
        self.kv_cache = DynamicCache(config=self.model.config)
        self.decode_chunk_size = 8  # just use default value here

    @property
    def device(self) -> torch.device:
        """Device where the model is located."""
        return self.model.device

    @property
    def num_cached_tokens(self) -> int:
        """Number of tokens currently stored in the KV cache."""
        return self.kv_cache.get_seq_length(0)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: DecodeMethod,
    ) -> torch.Tensor:
        """Generate new tokens given the context.

        KV cache are maintained internally across multiple calls.

        For multiple rounds of generation, `input_ids` in each subsequent call must be the
        full sequence of token IDs produced so far by this model (e.g., the output from the
        previous call, or a prefix of it). Passing a completely different or shorter sequence
        after generation has started will desynchronize the KV cache and cause incorrect behavior.

        Args:
            input_ids (torch.Tensor): Input context's token IDs of shape `[batch_size, seq_len]`.
            max_new_tokens (int): Limit on the number of new tokens to generate.
            decode_method (DecodeMethod): Token sampling strategy during generation.

        Returns:
            torch.Tensor: Updated token IDs. Shape `[batch_size, seq_len + num_new_tokens]`.

        Raises:
            ValueError: If `input_ids` length is less than or equal to the number of cached tokens.
            ValueError: If `decode_method` is unknown.
        """
        if max_new_tokens <= 0:
            return input_ids

        num_total_tokens = input_ids.shape[1]
        num_uncached_tokens = num_total_tokens - self.num_cached_tokens
        if num_uncached_tokens <= 0:
            raise ValueError(
                "The length of `input_ids` is less than or equal to the number of cached tokens. "
                "This usually indicates cache corruption or incorrect usage."
            )

        # Prefill
        if num_uncached_tokens > 1:
            prefill_out = self._prefill(
                input_ids=input_ids,
                num_uncached_tokens=num_uncached_tokens,
                decode_method=decode_method,
            )
            input_ids = prefill_out.output_ids
            max_new_tokens = max_new_tokens - 1

        # Decode
        if max_new_tokens > 0:
            input_ids = self._decode(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                decode_method=decode_method,
            )

        return input_ids

    @torch.no_grad()
    def _prefill(
        self,
        input_ids: torch.Tensor,
        num_uncached_tokens: int,
        decode_method: DecodeMethod,
        return_logits: bool = False,
    ) -> PrefillOutput:
        """Process the input token IDs in parallel and generate the next token.

        The new tokens (uncached tokens) in `input_ids` will be sliced for using as query tokens.

        KV cache is updated internally.
        The caller should make sure that `num_uncached_tokens > 0`, otherwise undefined behavior may happen.

        Return the updated `input_ids` (with one new token appended), and optionally the logits of
        the uncached tokens in `input_ids` (except the first token) + the logits of the new generated token,
        if `return_logits` is True.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape `[batch_size, seq_len]`.
            num_uncached_tokens (int): Number of uncached tokens in `input_ids`.
            decode_method (DecodeMethod): Token sampling strategy during prefill.
            return_logits (bool): Additionally return logits or not.

        Returns:
            PrefillOutput: Updated token IDs of shape `[batch_size, seq_len + 1]`,
                and optionally logits of shape `[batch_size, num_uncached_tokens, vocab_size]`.
                If `return_logits` is False, the returned logits will be an empty tensor.

        Raises:
            ValueError: If `decode_method` is unknown.
        """
        output_collector = OutputCollection(collect_logits=return_logits)

        output_ids, output_logits = self._forward(
            query_token_ids=input_ids[:, -num_uncached_tokens:],
            logits_to_keep=num_uncached_tokens,
            decode_method=decode_method,
        )
        output_collector.update(output_ids, output_logits)

        return PrefillOutput._make(output_collector.finalize())

    @torch.no_grad()
    def _decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: DecodeMethod,
        return_logits: bool = False,
    ) -> DecodeOutput:
        """Process input token IDs and generate new tokens, auto-regressively repeat.

        Stop when `max_new_tokens` is reached or an EOS token is generated.

        The last tokens in `input_ids` will be sliced for using as query tokens.

        KV cache is updated internally.
        The caller should make sure that the number of uncached tokens in `input_ids` is 1,
        otherwise undefined behavior may happen.

        Return the updated `input_ids` (with new decoded tokens appended), and optionally the logits of
        the new decoded tokens if `return_logits` is True.

        If `max_new_tokens <= 0`, no tokens will be generated, and the original `input_ids` will be returned,
        with a empty tensor as logits if `return_logits` is True.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape `[batch_size, seq_len]`
            max_new_tokens (int): Limit on the number of new tokens to generate.
            decode_method (DecodeMethod): Token sampling strategy during decoding.
            return_logits (bool): Additionally return logits of decoded tokens or not.

        Returns:
            DecodeOutput: Updated token IDs of shape `[batch_size, seq_len + num_new_tokens]`,
                and optionally logits of shape `[batch_size, num_new_tokens, vocab_size]`.
                If `return_logits` is False, the returned logits will be an empty tensor.

        Raises:
            ValueError: If `decode_method` is unknown.
        """
        output_collector = OutputCollection(collect_logits=return_logits)

        if max_new_tokens <= 0:
            output_collector.update(input_ids)
            return DecodeOutput._make(output_collector.finalize())

        num_new_tokens = 0
        max_chunks = (
            max_new_tokens + self.decode_chunk_size - 1
        ) // self.decode_chunk_size
        for _ in range(max_chunks):
            decode_chunk_size = min(
                self.decode_chunk_size, max_new_tokens - num_new_tokens
            )

            # 1. Decode `decode_chunk_size` tokens continuously
            for _ in range(decode_chunk_size):
                output_ids, output_logits = self._forward(
                    query_token_ids=input_ids[:, -1:],
                    logits_to_keep=1,
                    decode_method=decode_method,
                )
                output_collector.update(output_ids, output_logits)

                num_new_tokens += 1
                input_ids = output_ids

            # 2. Check for EOS token existence in the last chunk
            is_eos_token = (
                output_collector.output_ids[0, -decode_chunk_size:]
                == cast(int, self.model_config.eos_token_id)
            ).cpu()
            eos_found, eos_token_idx = torch.max(is_eos_token, dim=0)
            if eos_found.item():
                num_excess_tokens = decode_chunk_size - int(eos_token_idx.item()) - 1
                output_collector.trim(num_excess_tokens)
                break

        return DecodeOutput._make(output_collector.finalize())

    def _forward(
        self,
        query_token_ids: torch.Tensor,
        logits_to_keep: int,
        decode_method: DecodeMethod,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the query token IDs and generate the next token.

        Update the kv cache internally.

        Return the updated token IDs (with the ID of the newly generated token appended)
        and the computed logits.

        - If `logits_to_keep = 0`, return all computed logits, with shape
        `[batch_size, num_query_tokens, vocab_size]`.
        - If `logits_to_keep > 0`, only return the last `logits_to_keep` logits, with shape
        `[batch_size, logits_to_keep, vocab_size]`.

        Args:
            query_token_ids (torch.Tensor): Query token IDs of shape `[batch_size, num_query_tokens]`.
            logits_to_keep (int): Number of last logits to keep. If 0, keep all logits.
            decode_method (DecodeMethod)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated token IDs and computed logits.

        Raises:
            ValueError: If `decode_method` is unknown.
        """
        # 1. Forward
        # kv_cache is updated inside model.forward as a side effect
        forward_out = self.model.forward(
            input_ids=cast(torch.LongTensor, query_token_ids),
            logits_to_keep=logits_to_keep,
            use_cache=True,
            past_key_values=self.kv_cache,
        )

        # 2. Sample
        assert forward_out.logits is not None
        output_logits = forward_out.logits
        next_token_logits = output_logits[:, -1].to(dtype=torch.float32, copy=True)
        next_token_ids = self._sample(next_token_logits, decode_method)

        # 3. Update
        output_ids = torch.cat([query_token_ids, next_token_ids], dim=-1)

        return output_ids, output_logits

    def _sample(
        self, next_token_logits: torch.Tensor, decode_method: DecodeMethod
    ) -> torch.Tensor:
        """Sample next token IDs from logits according to `decode_method`.

        Args:
            next_token_logits (torch.Tensor): Logits of shape `[batch_size, vocab_size]`.
            decode_method (DecodeMethod)

        Returns:
            torch.Tensor: Sampled next token IDs of shape `[batch_size, 1]`.

        Raises:
            ValueError: If `decode_method` is unknown.
        """
        match decode_method:
            case "greedy":
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            case "random":
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1)
            case _:
                raise ValueError(f"Unknown decode method: {decode_method}")

        return next_token_ids

    def _reset(self) -> None:
        """Reset the model state for a new inference session.

        Primarily used for testing purpose.
        """
        self.kv_cache.crop(0)
        self.num_cached_tokens = 0

    def _print_kvcache_shape(self) -> None:
        """Print model's kv cache shape.

        Primarily used for debugging purpose.
        It is assumed that all layers have the same kv cache shape.
        """
        layer = cast(DynamicLayer, self.kv_cache.layers[0])
        if layer.keys is not None:
            print(f"Keys shape: {layer.keys.shape}", end=", ")
        else:
            print("Keys shape: None", end=", ")

        if layer.values is not None:
            print(f"Values shape: {layer.values.shape}")
        else:
            print("Values shape: None")
