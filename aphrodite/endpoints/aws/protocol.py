# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
import time
from typing import Dict, List, Literal, Optional, Union

from pydantic import (AliasChoices, BaseModel, Field, model_validator,
                      root_validator)
import torch

from aphrodite.common.utils import random_uuid
from aphrodite.common.sampling_params import SamplingParams


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{random_uuid()}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: str = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pygmalionai"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class CompletionRequest(BaseModel):
    model: str
    # a string, array of strings, array of tokens, or array of token arrays
    prompt: Union[List[int], List[List[int]], str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tfs: Optional[float] = 1.0
    eta_cutoff: Optional[float] = 0.0
    epsilon_cutoff: Optional[float] = 0.0
    typical_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    seed: Optional[int] = None
    include_stop_str_in_output: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    top_k: Optional[int] = -1
    top_a: Optional[float] = 0.0
    min_p: Optional[float] = 0.0
    mirostat_mode: Optional[int] = 0
    mirostat_tau: Optional[float] = 0.0
    mirostat_eta: Optional[float] = 0.0
    dynatemp_min: Optional[float] = Field(0.0,
                                          validation_alias=AliasChoices(
                                              "dynatemp_min", "dynatemp_low"),
                                          description="Aliases: dynatemp_low")
    dynatemp_max: Optional[float] = Field(0.0,
                                          validation_alias=AliasChoices(
                                              "dynatemp_max", "dynatemp_high"),
                                          description="Aliases: dynatemp_high")
    dynatemp_exponent: Optional[float] = 1.0
    smoothing_factor: Optional[float] = 0.0
    smoothing_curve: Optional[float] = 1.0
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    custom_token_bans: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    grammar: Optional[str] = None
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None

    def to_sampling_params(self) -> SamplingParams:
        echo_without_generation = self.echo and self.max_tokens == 0

        logits_processors = None
        if self.logit_bias:

            def logit_bias_logits_processor(
                    token_ids: List[int],
                    logits: torch.Tensor) -> torch.Tensor:
                for token_id, bias in self.logit_bias.items():
                    bias = min(100, max(-100, bias))
                    logits[int(token_id)] += bias
                return logits

            logits_processors = [logit_bias_logits_processor]
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_tokens if not echo_without_generation else 1,
            temperature=self.temperature,
            top_p=self.top_p,
            tfs=self.tfs,
            eta_cutoff=self.eta_cutoff,
            epsilon_cutoff=self.epsilon_cutoff,
            typical_p=self.typical_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            top_a=self.top_a,
            min_p=self.min_p,
            mirostat_mode=self.mirostat_mode,
            mirostat_tau=self.mirostat_tau,
            mirostat_eta=self.mirostat_eta,
            dynatemp_min=self.dynatemp_min,
            dynatemp_max=self.dynatemp_max,
            dynatemp_exponent=self.dynatemp_exponent,
            smoothing_factor=self.smoothing_factor,
            smoothing_curve=self.smoothing_curve,
            ignore_eos=self.ignore_eos,
            use_beam_search=self.use_beam_search,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs if self.echo else None,
            stop_token_ids=self.stop_token_ids,
            custom_token_bans=self.custom_token_bans,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            stop=self.stop,
            best_of=self.best_of,
            include_stop_str_in_output=self.include_stop_str_in_output,
            seed=self.seed,
            logits_processors=logits_processors,
        )

    @model_validator(mode="before")
    @classmethod
    def check_guided_decoding_count(cls, data):
        guide_count = sum([
            "guided_json" in data and data["guided_json"] is not None,
            "guided_regex" in data and data["guided_regex"] is not None,
            "guided_choice" in data and data["guided_choice"] is not None
        ])
        if guide_count > 1:
            raise ValueError(
                "You can only use one kind of guided decoding "
                "('guided_json', 'guided_regex' or 'guided_choice').")
        return data


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: Optional[List[Optional[Dict[int, float]]]] = None


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class Prompt(BaseModel):
    prompt: str
