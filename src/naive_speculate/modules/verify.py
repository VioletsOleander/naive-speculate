from typing import TypeAlias

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from naive_speculate.utils import Config

from .draft import Drafter

ModelOutputType: TypeAlias = GenerateDecoderOnlyOutput


class Verifier(Drafter):
    def __init__(self, config: Config):
        super().__init__(config)

    def _greedy_decode(
        self,
        draft_output: ModelOutputType,
        model_output: ModelOutputType,
    ) -> torch.Tensor:
        draft_ids = draft_output.sequences[0]
        model_ids = model_output.sequences[0]
        matches = draft_ids == model_ids
        verified_ids = model_ids[: torch.argmin(matches.to(torch.uint8))]
        return verified_ids[None, :]

    def _stochastic_decode(
        self,
        draft_output: ModelOutputType,
        model_output: ModelOutputType,
    ):
        assert draft_output.scores is not None
        proposal_distributions = torch.cat(draft_output.scores, dim=0).softmax(dim=-1)
        print(
            f"Proposal Distribution: {proposal_distributions}, shape: {proposal_distributions.shape}"
        )

        assert model_output.scores is not None
        num_new_tokens = proposal_distributions.shape[0]
        target_distribution = torch.cat(
            model_output.scores[-num_new_tokens:], dim=0
        ).softmax(dim=-1)

        print(
            f"Target Distribution: {target_distribution}, shape: {target_distribution.shape}"
        )

    def verify(self, draft: ModelOutputType, context: str) -> str:
        model_input = self.tokenize([context])
        model_output: ModelOutputType = self.model.generate(**model_input)  # type: ignore

        match self.config.decode_method:
            case "greedy":
                verified_ids = self._greedy_decode(draft, model_output)
            # case "stochastic":
            #     self._stochastic_decode(draft, model_output)

        verified_text = self.detokenize(verified_ids)[0]

        return verified_text

    def __str__(self) -> str:
        string = f"Verifier(model_name={self.config.verifier_model_name})\nType: {type(self.model)}\n{self.model}"
        return string
