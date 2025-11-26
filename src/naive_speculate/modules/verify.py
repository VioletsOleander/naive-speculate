import torch
from torch import Tensor
from transformers import BatchEncoding, GenerationConfig

from naive_speculate.utility import Config

from .draft import Drafter, ModelOutputType, ModelType, TokenizerType


class Verifier(Drafter):
    def __init__(self, config: Config):
        self.config = config

        self.model = ModelType.from_pretrained(
            config.verifier_model_name, device_map="auto", dtype="auto"
        )
        self._prepare_generation_config()

        self.tokenizer = TokenizerType.from_pretrained(config.verifier_model_name)
        self.tokenizer.padding_side = "left"

    def _greedy_decode(
        self,
        draft_output: ModelOutputType,
        model_output: ModelOutputType,
    ) -> torch.Tensor:
        """Perform greedy decoding verification.

        Returns:
            torch.Tensor: Verified token IDs. The shape is (1, sequence_length).
        """
        draft_ids = draft_output.sequences[0]
        model_ids = model_output.sequences[0]
        matches = draft_ids == model_ids[:-1]

        if matches.all():
            verified_ids = model_ids
        else:
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

    def verify(self, draft: ModelOutputType, model_input: BatchEncoding) -> Tensor:
        verify_config = GenerationConfig(
            max_new_tokens=1,
        )
        model_output: ModelOutputType = self.model.generate(
            **model_input, generation_config=verify_config, use_model_defaults=True  # type: ignore
        )
        match self.config.decode_method:
            case "greedy":
                verified_ids = self._greedy_decode(draft, model_output)
            # case "stochastic":
            #     self._stochastic_decode(draft, model_output)

        return verified_ids

    def __str__(self) -> str:
        string = f"Verifier(model_name={self.config.verifier_model_name})\nType: {type(self.model)}\n{self.model}"
        return string
