import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PretrainedConfig,
    AutoConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from os.path import isdir, exists, join
from dataclasses import dataclass
from typing import Optional
from safetensors.torch import load_file
from safetensors import safe_open
from collections import defaultdict
import json
from data.utils import IGNORE_INDEX

logger = logging.getLogger("model")


class DataSplitClassifier(nn.Module):
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "id": nn.Identity,
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size=100,
        num_hidden_layers=1,
        activation_str="id",
        bias=False,
    ):
        super().__init__()
        if activation_str not in DataSplitClassifier.activation_map:
            raise RuntimeError(
                f"Activation string {activation_str} not supported. Must be one of {DataSplitClassifier.activation_map.keys()}"
            )
        self.activation_str = activation_str
        act = self.get_activation_cls()

        if num_hidden_layers > 1 and self.activation_str == "id":
            raise RuntimeError(
                "Trying to set more than 1 hidden layer with identity activation is pointless"
            )

        layers = []
        # Down Proj layer
        layers.append(nn.Linear(input_dim, hidden_size, bias=bias))
        layers.append(act())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            layers.append(act())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_dim, bias=False))
        self.proj = nn.Sequential(*layers)

        logger.info(
            f"Initialized classifier head with {num_hidden_layers} hidden layers, hidden size {hidden_size}, and activation of type {self.get_activation_cls()}"
        )

    def get_activation_cls(self):
        return DataSplitClassifier.activation_map[self.activation_str]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (batch_size, seq_len, hidden_size)
        returns: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Flatten batch and seq_len, apply the module, then reshape back
        out = self.proj(hidden_states.view(-1, hidden_size))
        out = out.view(batch_size, seq_len, -1)  # (batch_size, seq_len, vocab_size)
        return out

    def target_token_logits(
        self, hidden_states: torch.Tensor, target_token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes classifier logits only for selected target token ids at each position.

        hidden_states: (batch_size, seq_len, hidden_size)
        target_token_ids: (batch_size, seq_len)
        returns: (batch_size, seq_len)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        if target_token_ids.shape != (batch_size, seq_len):
            raise RuntimeError(
                f"Expected target_token_ids shape {(batch_size, seq_len)}, got {tuple(target_token_ids.shape)}"
            )

        x = hidden_states.reshape(-1, hidden_size)
        for layer in self.proj[:-1]:
            x = layer(x)

        final_layer = self.proj[-1]
        if target_token_ids.dtype != torch.long:
            target_token_ids = target_token_ids.to(torch.long)
        flat_token_ids = target_token_ids.reshape(-1)

        selected_weights = final_layer.weight.index_select(0, flat_token_ids)
        selected_logits = (x * selected_weights).sum(dim=-1)
        if final_layer.bias is not None:
            selected_logits = selected_logits + final_layer.bias.index_select(
                0, flat_token_ids
            )
        return selected_logits.view(batch_size, seq_len)


@dataclass
class T3CausalLMOutputWithPast(CausalLMOutputWithPast):
    classifier_logits: Optional[torch.FloatTensor] = None
    base_logits: Optional[torch.FloatTensor] = None


class T3CausalLMConfig(PretrainedConfig):
    model_type = "t3_causal_lm"

    def __init__(
        self,
        guidance_kwargs=None,
        pooling="mean",
        pool_temp=None,
        extraction_layer=-1,
        guidance_scale=1,
        base_temp=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_config_dict = kwargs.copy()
        self.guidance = guidance_kwargs or {}
        self.pooling = pooling
        self.pool_temp = pool_temp
        self.extraction_layer = extraction_layer
        self.base_temp = base_temp
        self.guidance_scale = guidance_scale

    def to_base(self):
        base_config_dict = self.base_config_dict.copy()
        base_model_type = base_config_dict.pop("model_type")
        return AutoConfig.for_model(base_model_type, **base_config_dict)


def _mean_pool(hidden_states, attention_mask=None):
    if attention_mask is None:
        _, seq_len, _ = hidden_states.shape  # batch x seq_len x hidden_size
        return hidden_states.cumsum(dim=1) / torch.arange(
            1, seq_len + 1, device=hidden_states.device
        ).view(1, -1, 1)
    mask = attention_mask.to(
        device=hidden_states.device, dtype=hidden_states.dtype
    ).unsqueeze(-1)
    token_counts = mask.cumsum(dim=1).clamp_min(1)
    return (hidden_states * mask).cumsum(dim=1) / token_counts


class T3CausalLM(PreTrainedModel):
    """
    Wraps a pretrained CausalLM, freezing all base model weights,
    and adds a DataSplitDiscriminator to produce a logit adjustment vector.

    Training: Only the classifier is updated, with loss based on retain/forget classification.
    Inference: Base model logits + adjustment logits are used for token prediction.
    """

    config_class = T3CausalLMConfig
    pooling_fn_dict = {
        "mean": _mean_pool,
    }

    def __init__(self, config: T3CausalLMConfig, base_lm=None):
        super().__init__(config)

        if base_lm is None:
            base_config = config.to_base()
            self.base_lm = AutoModelForCausalLM.from_config(base_config)
        else:
            self.base_lm = base_lm

        if not hasattr(self.base_lm, "generation_config"):
            logger.warning(
                "Could not find base model generation config, resorting to default"
            )
            self.base_lm.generation_config = GenerationConfig()

        self.generation_config = self.base_lm.generation_config

        # Freeze base model and set to eval
        self.base_lm.eval()
        for p in self.base_lm.parameters():
            p.requires_grad_(False)

        classifier_args = config.guidance.copy()

        base_device = next(self.base_lm.parameters()).device
        base_dtype = next(self.base_lm.parameters()).dtype

        self.guidance_head = DataSplitClassifier(
            input_dim=self.config.hidden_size,
            output_dim=self.config.vocab_size,
            **classifier_args,
        ).to(device=base_device, dtype=base_dtype)

        assert (
            config.pooling in T3CausalLM.pooling_fn_dict.keys()
        ), f"Pooling function string {config.pooling} not recognized"
        self.pooling_fn = T3CausalLM.pooling_fn_dict[config.pooling]
        self.pooling_fn_name = config.pooling
        self.pool_temp = config.pool_temp

        self.base_lm.config.output_hidden_states = True
        self.base_lm.config.output_attentions = self.pooling_fn_name == "attn"

        self.extraction_layer = config.extraction_layer
        self.guidance_scale = config.guidance_scale
        self.base_temp = config.base_temp

        # Set attn support
        base_lm_attn_implementation = (
            self.base_lm.__class__._autoset_attn_implementation(
                self.base_lm.config,
                torch_dtype=self.base_lm.config.torch_dtype,
                device_map=None,
            )._attn_implementation
        )
        self.config.attn_implementation = base_lm_attn_implementation

        self.lm_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        logger.info(
            f"Initialized a T3CausalLM model:\n"
            f"base_lm: {self.base_lm.config._name_or_path}\n"
            f"pooling: {config.pooling}\n"
            f"pool_temp: {config.pool_temp}\n"
            f"extraction layer: {self.extraction_layer}\n"
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        assert isdir(
            pretrained_model_name_or_path
        ), "Tried to load T3CausalLM but did not pass a valid path to saved model"

        try:
            config_path = join(pretrained_model_name_or_path, "config.json")
            logger.info(f"Found saved config at {config_path}")
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            guidance_kwargs = config_dict.pop("guidance")
            base_kwargs = config_dict.pop("base_config_dict")
            pooling = config_dict["pooling"]
            pool_temp = config_dict["pool_temp"]
            guidance_scale = config_dict["guidance_scale"]
            base_temp = config_dict.get("base_temp", 1.0)
            extraction_layer = config_dict["extraction_layer"]
            config = T3CausalLMConfig(
                guidance_kwargs=guidance_kwargs,
                pooling=pooling,
                pool_temp=pool_temp,
                extraction_layer=extraction_layer,
                guidance_scale=guidance_scale,
                base_temp=base_temp,
                **base_kwargs,
            )
        except Exception:
            logger.exception(
                "Failed to load config.json from %s", pretrained_model_name_or_path
            )
            raise

        try:
            model_path_single = join(pretrained_model_name_or_path, "model.safetensors")
            if exists(model_path_single):
                logger.info(f"Loading model from single file {model_path_single}")
                state_dict = load_file(model_path_single)
            else:
                logger.info(
                    f"Couldn't find single model.safetensors file at path {pretrained_model_name_or_path}. Attempting to look for sharded files."
                )
                state_dict = defaultdict(dict)
                index_file = join(
                    pretrained_model_name_or_path, "model.safetensors.index.json"
                )
                if exists(index_file):
                    with open(index_file, "r") as f:
                        index = json.load(f)
                    for key, fname in index["weight_map"].items():
                        shard_path = join(pretrained_model_name_or_path, fname)
                        with safe_open(shard_path, framework="pt", device="cpu") as f:
                            state_dict[key] = f.get_tensor(key)
                else:
                    raise FileNotFoundError(
                        f"No safetensors file found in {pretrained_model_name_or_path}"
                    )

        except Exception as e:
            logger.error(
                f"Failed to load safetensors, trying pytorch bin files. Error {e}"
            )
            state_dict = torch.load(
                join(pretrained_model_name_or_path, "pytorch_model.bin"),
                map_location="cpu",
            )

        base_config = config.to_base()
        kwargs.pop("config", None)

        base_lm = AutoModelForCausalLM.from_pretrained(
            base_config._name_or_path, *args, config=base_config, **kwargs
        )

        base_lm_state_dict = {
            k.replace("base_lm.", ""): v
            for k, v in state_dict.items()
            if k.startswith("base_lm.")
        }
        incomp = base_lm.load_state_dict(base_lm_state_dict, strict=False)
        if incomp.missing_keys or incomp.unexpected_keys:
            logger.warning(
                f"Some weights of the base_lm could not be loaded.\n"
                f"Missing keys: {incomp.missing_keys}\n"
                f"Unexpected keys: {incomp.unexpected_keys}"
            )

            for missing_key in incomp.missing_keys:
                missing_weight = base_lm.state_dict()[missing_key]
                for n, p in base_lm.named_parameters():
                    if n != missing_key and p.data_ptr() == missing_weight.data_ptr():
                        logger.info(
                            f"Missing key {missing_key} is tied to {n}. If {n} is loaded this will be fixed by tie_weights()."
                        )

        base_lm.tie_weights()

        model = cls(config, base_lm=base_lm)
        guidance_state_dict = {
            k.replace("guidance_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("guidance_head.")
        }
        model.guidance_head.load_state_dict(guidance_state_dict)
        return model

    @classmethod
    def from_pretrained_base(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        guidance_kwargs=None,
        pooling="mean",
        pool_temp=None,
        extraction_layer=-1,
        guidance_scale=1,
        base_temp=1,
        **kwargs,
    ):
        base_lm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        return cls.from_pretrained_base_obj(
            base_lm=base_lm,
            guidance_kwargs=guidance_kwargs,
            pooling=pooling,
            pool_temp=pool_temp,
            extraction_layer=extraction_layer,
            guidance_scale=guidance_scale,
            base_temp=base_temp,
        )

    @classmethod
    def from_pretrained_base_obj(
        cls,
        base_lm,
        guidance_kwargs=None,
        pooling="mean",
        pool_temp=None,
        extraction_layer=-1,
        guidance_scale=1,
        base_temp=1,
    ):
        config = T3CausalLMConfig(
            guidance_kwargs=guidance_kwargs,
            pooling=pooling,
            pool_temp=pool_temp,
            extraction_layer=extraction_layer,
            guidance_scale=guidance_scale,
            base_temp=base_temp,
            **base_lm.config.to_dict(),
        )
        return cls(config, base_lm=base_lm)

    def can_generate(self):
        return True

    def generate(self, inputs=None, **kwargs):
        if inputs is None:
            inputs = kwargs.pop("input_ids", None)
        return self.custom_generate(inputs, self.generation_config, **kwargs)

    @torch.no_grad()
    def custom_generate(self, input_ids, generation_config, **kwargs):
        if self.pooling_fn_name != "mean":
            raise NotImplementedError(
                "T3 custom generation currently supports only pooling='mean'."
            )
        if input_ids is None:
            raise ValueError("input_ids must be provided for T3 custom generation.")

        def get_gen_param(name, default=None):
            if name in kwargs:
                return kwargs.pop(name)
            return getattr(generation_config, name, default)

        do_sample = get_gen_param("do_sample", default=False)
        top_p = get_gen_param("top_p", default=None)
        if top_p is not None:
            raise NotImplementedError("T3 custom generation does not support top_p.")
        temperature = get_gen_param("temperature", default=None)
        if do_sample:
            if temperature is None:
                temperature = 1.0
            if temperature <= 0:
                raise ValueError("T3 sampling requires temperature > 0.")

        num_beams = get_gen_param("num_beams", default=1)
        if num_beams not in (None, 1):
            raise NotImplementedError(
                "T3 custom generation does not support beam search."
            )

        num_return_sequences = get_gen_param("num_return_sequences", default=1)
        if num_return_sequences not in (None, 1):
            raise NotImplementedError(
                "T3 custom generation supports only one returned sequence."
            )

        if get_gen_param("return_dict_in_generate", default=False):
            raise NotImplementedError(
                "T3 custom generation returns only the generated token tensor."
            )

        max_new_tokens = get_gen_param("max_new_tokens", default=None)
        if max_new_tokens is None:
            raise ValueError(
                "max_new_tokens must be specified for T3 custom generation."
            )

        use_cache = get_gen_param("use_cache", default=True)
        attention_mask = get_gen_param("attention_mask", default=None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device=input_ids.device)

        eos_token_id = get_gen_param("eos_token_id", default=None)
        if eos_token_id is not None:
            if torch.is_tensor(eos_token_id):
                eos_token_id = eos_token_id.to(
                    device=input_ids.device, dtype=torch.long
                ).flatten()
            elif isinstance(eos_token_id, (list, tuple)):
                eos_token_id = torch.tensor(
                    eos_token_id, dtype=torch.long, device=input_ids.device
                ).flatten()
            else:
                eos_token_id = torch.tensor(
                    [eos_token_id], dtype=torch.long, device=input_ids.device
                )
        pad_token_id = get_gen_param("pad_token_id", default=None)
        if pad_token_id is None:
            pad_token_id = (
                int(eos_token_id[0].item()) if eos_token_id is not None else 0
            )
        if torch.is_tensor(pad_token_id):
            pad_token_id = int(pad_token_id.flatten()[0].item())

        stopping_criteria = get_gen_param("stopping_criteria", default=None)

        if kwargs:
            unsupported = ", ".join(sorted(kwargs.keys()))
            raise NotImplementedError(
                f"T3 custom generation does not support these generation kwargs: {unsupported}"
            )

        if use_cache:
            return self._custom_generate_with_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                stopping_criteria=stopping_criteria,
                do_sample=do_sample,
                temperature=temperature,
            )
        return self.custom_generate_no_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            stopping_criteria=stopping_criteria,
            do_sample=do_sample,
            temperature=temperature,
        )

    @torch.no_grad()
    def custom_generate_no_cache(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        eos_token_id=None,
        pad_token_id=0,
        stopping_criteria=None,
        do_sample=False,
        temperature=None,
    ):
        generated = input_ids
        full_attention_mask = attention_mask
        batch_size = generated.shape[0]
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=generated.device
        )

        for _ in range(max_new_tokens):
            base_outputs = self.base_lm(
                input_ids=generated,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
            extracted_states = base_outputs.hidden_states[self.extraction_layer]
            pooled_states = self.pooling_fn(extracted_states, full_attention_mask)
            classifier_logits = self.guidance_head(pooled_states[:, -1:, :])
            guided_logits = self._guide_logits(
                base_outputs.logits[:, -1:, :], classifier_logits
            ).squeeze(1)
            if do_sample:
                probs = torch.softmax(guided_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(guided_logits, dim=-1)
            pad_tokens = torch.full_like(next_tokens, int(pad_token_id))
            next_tokens = torch.where(
                unfinished_sequences.bool(), next_tokens, pad_tokens
            )
            next_attention = unfinished_sequences.to(full_attention_mask.dtype)

            generated = torch.cat([generated, next_tokens[:, None]], dim=-1)
            full_attention_mask = torch.cat(
                [full_attention_mask, next_attention[:, None]], dim=-1
            )

            if eos_token_id is not None:
                is_eos = torch.isin(next_tokens, eos_token_id)
                unfinished_sequences = unfinished_sequences * (~is_eos).long()

            stopping_done = False
            if stopping_criteria is not None:
                stopping_result = stopping_criteria(generated, guided_logits)
                stopping_done = (
                    bool(stopping_result.all().item())
                    if torch.is_tensor(stopping_result)
                    else bool(stopping_result)
                )
            if unfinished_sequences.max() == 0 or stopping_done:
                break

        return generated

    @torch.no_grad()
    def _custom_generate_with_cache(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        eos_token_id=None,
        pad_token_id=0,
        stopping_criteria=None,
        do_sample=False,
        temperature=None,
    ):
        generated = input_ids
        full_attention_mask = attention_mask
        batch_size = generated.shape[0]
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=generated.device
        )
        past_key_values = None
        current_input_ids = input_ids
        current_token_mask = attention_mask
        hidden_sum = None
        token_count = None

        for _ in range(max_new_tokens):
            base_outputs = self.base_lm(
                input_ids=current_input_ids,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True,
                return_dict=True,
            )
            extracted_states = base_outputs.hidden_states[self.extraction_layer]
            mask = current_token_mask.to(
                device=extracted_states.device, dtype=extracted_states.dtype
            )
            step_hidden_sum = (extracted_states * mask.unsqueeze(-1)).sum(dim=1)
            step_token_count = mask.sum(dim=1, keepdim=True)

            if hidden_sum is None:
                if (step_token_count == 0).any():
                    raise ValueError(
                        "Cannot generate from a prompt with no non-padding tokens."
                    )
                hidden_sum = step_hidden_sum
                token_count = step_token_count
            else:
                hidden_sum = hidden_sum + step_hidden_sum
                token_count = token_count + step_token_count

            pooled_states = hidden_sum / token_count.clamp_min(1)
            classifier_logits = self.guidance_head(pooled_states.unsqueeze(1))
            guided_logits = self._guide_logits(
                base_outputs.logits[:, -1:, :], classifier_logits
            ).squeeze(1)
            if do_sample:
                probs = torch.softmax(guided_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(guided_logits, dim=-1)
            pad_tokens = torch.full_like(next_tokens, int(pad_token_id))
            next_tokens = torch.where(
                unfinished_sequences.bool(), next_tokens, pad_tokens
            )
            next_attention = unfinished_sequences.to(full_attention_mask.dtype)

            generated = torch.cat([generated, next_tokens[:, None]], dim=-1)
            full_attention_mask = torch.cat(
                [full_attention_mask, next_attention[:, None]], dim=-1
            )

            past_key_values = base_outputs.past_key_values
            current_input_ids = next_tokens[:, None]
            current_token_mask = next_attention[:, None]

            if eos_token_id is not None:
                is_eos = torch.isin(next_tokens, eos_token_id)
                unfinished_sequences = unfinished_sequences * (~is_eos).long()

            stopping_done = False
            if stopping_criteria is not None:
                stopping_result = stopping_criteria(generated, guided_logits)
                stopping_done = (
                    bool(stopping_result.all().item())
                    if torch.is_tensor(stopping_result)
                    else bool(stopping_result)
                )
            if unfinished_sequences.max() == 0 or stopping_done:
                break

        return generated

    def _guide_logits(self, base_logits, classifier_logits):
        """
        base_logits: (batch, seq_len, vocab)
        classifier_logits: (batch, seq_len, vocab)
        """

        base_log_probs = F.log_softmax(base_logits, dim=2)  # batch x seq_len x vocab

        # Consider clipping this for stability
        classifier_log_probs = F.logsigmoid(classifier_logits)

        return (
            base_log_probs / self.base_temp + self.guidance_scale * classifier_log_probs
        )

    def forward(
        self, input_ids=None, attention_mask=None, classifier_only=False, **kwargs
    ):
        if classifier_only:
            labels = kwargs.pop("labels", None)
            if labels is None:
                raise ValueError("labels must be passed when classifier_only=True")
            if not hasattr(self.base_lm, "get_decoder"):
                raise NotImplementedError(
                    "classifier_only mode not implemented for base models without a get_decoder() method"
                )

            if self.extraction_layer != -1:
                raise NotImplementedError(
                    "classifier_only mode with extraction_layer != -1 not implemented, as it would require replicating part of the base LM forward pass. Set extraction_layer to -1 to use the final hidden states for classification."
                )

            decoder = self.base_lm.get_decoder()
            with torch.no_grad():
                dec_out = decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
            # (batch, seq_len, hidden_size)
            extracted_states = dec_out.last_hidden_state
            pooled_states = self.pooling_fn(extracted_states, attention_mask)
            shifted_labels = labels[:, 1:].contiguous()
            # Assign token 0 to IGNORE_INDEX positions to avoid out of bounds error
            safe_shifted_labels = shifted_labels.masked_fill(
                shifted_labels == IGNORE_INDEX, 0
            )
            # (batch, seq_len-1)
            classifier_logits = self.guidance_head.target_token_logits(
                pooled_states[:, :-1, :].contiguous(),
                safe_shifted_labels,
            )
            return T3CausalLMOutputWithPast(classifier_logits=classifier_logits)

        else:
            _ = kwargs.pop("labels", None)
            kwargs["output_hidden_states"] = True
            kwargs["output_attentions"] = False
            base_outputs = self.base_lm(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

            # Adjust logits using the classifier guidance
            # (batch, seq_len, hidden_size)
            extracted_states = base_outputs.hidden_states[self.extraction_layer]
            pooled_states = self.pooling_fn(extracted_states, attention_mask)

            # (batch, seq_len, vocab)
            classifier_logits = self.guidance_head(pooled_states)

            with torch.no_grad():
                guided_logits = self._guide_logits(
                    base_outputs.logits, classifier_logits
                )

            outputs = T3CausalLMOutputWithPast(
                loss=None,
                logits=guided_logits,
                past_key_values=base_outputs.past_key_values,
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
                base_logits=base_outputs.logits,
                classifier_logits=classifier_logits,
            )
            return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        self.guidance_head.train(mode)
        # always keep base model in eval mode
        self.base_lm.eval()
        return self

    def eval(self):
        super().eval()
        self.guidance_head.eval()
        self.base_lm.eval()
        return self


CONFIG_MAPPING.register("t3_causal_lm", T3CausalLMConfig)
MODEL_FOR_CAUSAL_LM_MAPPING.register(T3CausalLMConfig, T3CausalLM)
