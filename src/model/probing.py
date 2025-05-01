from transformers import AutoConfig, LlamaForCausalLM
import torch
import torch.nn as nn
import logging
import gc
from copy import deepcopy
from transformers import AutoModelForCausalLM

logger = logging.getLogger("model")


class ProbedLlamaForCausalLM(LlamaForCausalLM):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, n_layers: int = 100, **kwargs
    ):
        config, unused_kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
        )
        model: LlamaForCausalLM = super().from_pretrained(
            pretrained_model_name_or_path, config=config, **unused_kwargs
        )
        retain_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained('open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90')
        n_layers = min(n_layers, model.config.num_hidden_layers)
        model.config.num_hidden_layers = n_layers
        model.model.layers = nn.ModuleList(model.model.layers[:n_layers])
        model.lm_head = deepcopy(retain_model.lm_head)
        gc.collect()
        torch.cuda.empty_cache()
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("lm_head")
        logger.info(
            f"Initialised a ProbedLlamaForCausalLM model with {n_layers} layers"
        )
        return model
