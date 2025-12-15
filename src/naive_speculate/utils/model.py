from typing import Literal, cast, overload

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM, TextStreamer
from transformers.modeling_outputs import CausalLMOutputWithPast


class QwenModel:
    """Wrapper class for Qwen3ForCausalLM to simplify inference API and typing annotations.

    Attributes:
        model (Qwen3ForCausalLM): The Qwen model for causal language modeling.
        model_config (Qwen3Config): Configuration of the Qwen model.
        kv_cache (DynamicCache): Key-value cache for efficient generation.
    """

    model: Qwen3ForCausalLM
    model_config: Qwen3Config
    kv_cache: DynamicCache

    def __init__(self, model_name: str) -> None:
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )
        self.model_config = self.model.config
        self.kv_cache = DynamicCache(config=self.model.config)

    @overload
    def prefill(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: Literal[False] = False,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor: ...

    @overload
    def prefill(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: Literal[True],
        streamer: TextStreamer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def prefill(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: bool,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: bool = False,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Prefill the model with input_ids and generate one new token.

        Returns the updated input_ids (with one new token appended), and optionally the logits corresponding
        to the context (except the first token) and the generated one token if output_logits is True.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.
            decode_method (str): Decoding method, either "greedy" or "random".
            output_logits (bool): Whether to return logits along with generated tokens.
            streamer (TextStreamer | None): Optional streamer for output tokens.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Generated token IDs, and optionally logits.
        """
        if max_new_tokens <= 0:
            if not output_logits:
                return input_ids
            return input_ids, torch.empty(0, device=self.device)

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

        if not output_logits:
            return input_ids

        assert outputs.logits is not None
        return input_ids, outputs.logits

    @overload
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: Literal[False] = False,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor: ...

    @overload
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: Literal[True],
        streamer: TextStreamer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: bool,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        output_logits: bool = False,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode new tokens auto-regressively.

        Returns the updated input_ids (with new tokens appended), and optionally the logits corresponding to
        the decoded tokens if output_logits is True.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.
            decode_method (str): Decoding method, either "greedy" or "random".
            output_logits (bool): Whether to return logits along with generated tokens.
            streamer (TextStreamer | None): Optional streamer for output tokens.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Generated token IDs, and optionally logits.
        """
        if max_new_tokens <= 0:
            if not output_logits:
                return input_ids
            return input_ids, torch.empty(0, device=self.device)

        num_new_tokens = 0
        logits = []
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

            if output_logits:
                assert outputs.logits is not None
                logits.append(outputs.logits)

            if (
                num_new_tokens >= max_new_tokens
                or next_token_ids[0, 0].item() == self.model_config.eos_token_id
            ):
                break

        if not output_logits:
            return input_ids
        else:
            # the huggingface generate API returns tuple of tensors directly
            # here, we additionally concatenate along seq_len dimension to maintain API consistency with prefill
            # but notice additional cost with tensor allocation and copy is incurred
            return input_ids, torch.cat(logits, dim=1)

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

    def inference(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        decode_method: str,
        streamer: TextStreamer | None = None,
    ) -> torch.Tensor:
        input_ids = self.prefill(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            decode_method=decode_method,
            streamer=streamer,
        )
        output_ids = self.decode(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens - 1,  # prefill already generated 1 token
            decode_method=decode_method,
            streamer=streamer,
        )

        if streamer is not None:
            streamer.end()
        return output_ids

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __str__(self) -> str:
        return str(self.model)
