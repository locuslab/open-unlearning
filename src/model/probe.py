from transformers import AutoConfig, LlamaForCausalLM
import torch
import torch.nn as nn
import logging
import gc
from copy import deepcopy
from transformers import AutoModelForCausalLM

logger = logging.getLogger("model")


class ProbedLlamaForCausalLM(LlamaForCausalLM):
    """
    Class for loading a LlamaForCausalLM model with the following custom behavior:
    - Initializes only the first `n_layers` of the model.
    - Sets up a newly initialized `lm_head`, optionally using weights from
     `head_pretrained_model_name_or_path`
    - Trains only the lm_head parameters with rest of the model frozen.
    - Once the model is saved during training, for inference it can also be loaded using
      AutoModelForCausalLM
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        head_pretrained_model_name_or_path: str = None,
        n_layers: int = 100,
        freeze_base_model: bool = True,
        **kwargs,
    ):
        config, unused_kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
        )
        config.tie_word_embeddings = False
        model: LlamaForCausalLM = super().from_pretrained(
            pretrained_model_name_or_path, config=config, **unused_kwargs
        )

        # Limit number of transformer layers
        n_layers = min(n_layers, model.config.num_hidden_layers)
        model.config.num_hidden_layers = n_layers
        model.model.layers = nn.ModuleList(model.model.layers[:n_layers])

        # Reinitialize lm_head
        ref_params = list(model.model.layers[-1].parameters())[0]
        device = ref_params.device
        if head_pretrained_model_name_or_path is not None:
            logger.info(
                f"Initialising lm_head from {head_pretrained_model_name_or_path}"
            )
            head_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
                head_pretrained_model_name_or_path, config=config, **unused_kwargs
            )
            lm_head = deepcopy(head_model.lm_head).to(device)
            model.set_output_embeddings(lm_head)
        else:
            logger.info("Initialising new lm_head")
            model._init_weights(model.lm_head)

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Set trainable params
        for name, p in model.named_parameters():
            p.requires_grad = not freeze_base_model or name.startswith("lm_head")
        logger.info(
            f"Initialised a ProbedLlamaForCausalLM model with {n_layers} layers"
        )
        return model
