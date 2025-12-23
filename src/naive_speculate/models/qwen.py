from typing import Literal, cast, overload

import torch
from transformers import (
    DynamicCache,
    DynamicLayer,
    Qwen3Config,
    Qwen3ForCausalLM,
)


class QwenModel:
    """Wrapper class for Qwen3ForCausalLM to simplify inference API and typing annotations.

    Currently only supports batch_size = 1. Maybe later on I can add batch support, and further
    including continuous batching, pd disaggregation, etc.

    Attributes:
        model (Qwen3ForCausalLM)
        model_config (Qwen3Config)
        kv_cache (DynamicCache)
        prefill_done (bool)
        decode_chunk_size (int): EOS token check interval during decoding, default to 8. Used as
        a simple trick to reduce device synchronization overhead.
    """

    model: Qwen3ForCausalLM
    model_config: Qwen3Config
    kv_cache: DynamicCache
    prefill_done: bool
    decode_chunk_size: int

    def __init__(self, model_name: str) -> None:
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )
        self.model_config = self.model.config
        self.kv_cache = DynamicCache(config=self.model.config)
        self.prefill_done = False
        self.decode_chunk_size = 8  # just use default value here

    @property
    def device(self) -> torch.device:
        """Device where the model is located."""
        return self.model.device

    def inference(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
    ) -> torch.Tensor:
        """Generate new tokens given the context.

        Args:
            input_ids (torch.Tensor): Input context's token IDs of shape `[batch_size, seq_len]`.
            max_new_tokens (int): Limit on the number of new tokens to generate.
            decode_method (str): Either "greedy" or "random".

        Returns:
            torch.Tensor: Updated token IDs. Shape `[batch_size, seq_len + num_new_tokens]`.
        """
        if max_new_tokens <= 0:
            return input_ids

        if not self.prefill_done:
            input_ids = self._prefill(
                input_ids=input_ids,
                decode_method=decode_method,
            )
            max_new_tokens = max_new_tokens - 1
            self.prefill_done = True

        output_ids = self._decode(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            decode_method=decode_method,
        )

        return output_ids

    @overload
    def _prefill(
        self,
        input_ids: torch.Tensor,
        decode_method: str,
        output_logits: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def _prefill(
        self,
        input_ids: torch.Tensor,
        decode_method: str,
        output_logits: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def _prefill(
        self,
        input_ids: torch.Tensor,
        decode_method: str,
        output_logits: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def _prefill(
        self,
        input_ids: torch.Tensor,
        decode_method: str,
        output_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Prefill the model with `input_ids` and generate one new token.

        Return the updated `input_ids` (with one new token appended), and optionally the logits of
        the context (except the first token) + the logits of the one new token, if `output_logits` is True.

        Args:
            input_ids (torch.Tensor): Input context's token IDs of shape `[batch_size, seq_len]`.
            decode_method (str): Either "greedy" or "random".
            output_logits (bool): Additionally return logits or not.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Updated token IDs of shape `[batch_size, seq_len + 1]`,
              and optionally logits of shape `[batch_size, seq_len, vocab_size]`.

        Raises:
            ValueError: If `decode_method` is unknown.
        """
        # 1. Forward
        # kv_cache is updated inside model.forward as a side effect
        outputs = self.model.forward(
            input_ids=cast(torch.LongTensor, input_ids),
            logits_to_keep=0,  # output all logits
            use_cache=True,
            past_key_values=self.kv_cache,
        )
        assert outputs.logits is not None

        # 2. Sample
        # copy to avoid keeping a hanging ref to full outputs.logits
        next_token_logits = outputs.logits[:, -1, :].to(dtype=torch.float32, copy=True)
        next_token_ids = self._sample(next_token_logits, decode_method)

        # 3. Update
        input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

        if not output_logits:
            return input_ids
        else:
            return input_ids, outputs.logits

    @overload
    def _decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def _decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def _decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def _decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode new tokens auto-regressively.

        Should be called after `_prefill`.
        Returns the updated `input_ids` (with new tokens appended), and optionally the logits of
        the decoded tokens if `output_logits` is True.

        Args:
            input_ids (torch.Tensor): Input context's token IDs of shape `[batch_size, seq_len]`.
            max_new_tokens (int): Limit on the number of new tokens to generate.
            decode_method (str): Either "greedy" or "random".
            output_logits (bool): Additionally return logits of decoded tokens or not.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Update token IDs of shape `[batch_size, seq_len + num_new_tokens]`,
              and optionally logits of shape `[batch_size, num_new_tokens, vocab_size]`.

        Raises:
            ValueError: If `decode_method` is unknown.
        """
        if max_new_tokens <= 0:
            if not output_logits:
                return input_ids
            else:
                return input_ids, torch.empty(0, device=self.device)

        logits = []
        max_chunks = (
            max_new_tokens + self.decode_chunk_size - 1
        ) // self.decode_chunk_size
        num_new_tokens = 0
        for _ in range(max_chunks):
            decode_chunk_size = min(
                self.decode_chunk_size, max_new_tokens - num_new_tokens
            )
            for _ in range(decode_chunk_size):
                # 1. Forward
                # kv_cache is updated inside model.forward as a side effect
                outputs = self.model.forward(
                    input_ids=cast(torch.LongTensor, input_ids[:, -1:]),
                    logits_to_keep=1,
                    use_cache=True,
                    past_key_values=self.kv_cache,
                )
                assert outputs.logits is not None

                # 2. Sample
                next_token_logits = outputs.logits[:, -1, :].to(dtype=torch.float32)
                next_token_ids = self._sample(next_token_logits, decode_method)

                # 3. Update
                input_ids = torch.cat([input_ids, next_token_ids], dim=-1)

                num_new_tokens += 1
                if output_logits:
                    logits.append(outputs.logits)

            # check for EOS token existance in the last chunk
            eos_token_id = cast(int, self.model_config.eos_token_id)
            is_eos = input_ids[0, -decode_chunk_size:] == eos_token_id
            eos_found, eos_token_idx = torch.max(is_eos, dim=0)
            if eos_found.item():
                input_ids = input_ids[:, : -(decode_chunk_size - eos_token_idx - 1)]
                break

        if not output_logits:
            return input_ids
        else:
            return input_ids, torch.cat(logits, dim=1)

    def _sample(
        self, next_token_logits: torch.Tensor, decode_method: str
    ) -> torch.Tensor:
        """Sample next token IDs from logits according to `decode_method`.

        Args:
            next_token_logits (torch.Tensor): Logits of shape `[batch_size, vocab_size]`.
            decode_method (str): Either "greedy" or "random".

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
        self.prefill_done = False

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
