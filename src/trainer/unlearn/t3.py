import logging

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from trainer.unlearn.base import UnlearnTrainer
from model.t3 import T3CausalLM
from data.utils import IGNORE_INDEX

logger = logging.getLogger("trainer")


class T3(UnlearnTrainer):
    def __init__(
        self,
        pooling="mean",
        pool_temp=None,
        extraction_layer=-1,
        guidance_scale=1,
        base_temp=1,
        guidance_cfg=None,
        *args,
        **kwargs,
    ):

        # Extract the base model from kwargs before super init, call init using the wrapped model
        if "model" in kwargs:
            base_lm = kwargs.pop("model")
        else:
            # Also check if model was passed as positional argument
            if len(args) > 0:
                base_lm = args[0]
                args = args[1:]  # Remove model from args
            else:
                raise Exception("Couldn't parse model - no model provided")
        
        self.guidance_kwargs = OmegaConf.to_container(guidance_cfg, resolve=True)
        self.pooling = pooling
        self.pool_temp = pool_temp
        if self.pool_temp is not None and self.pooling != "attn":
            raise RuntimeError(f"Attempting to set pool_temp for pooling function {pooling}, but pool_temp is only supported for attn pooling.")
        self.extraction_layer = extraction_layer
        self.guidance_scale=guidance_scale
        self.base_temp=base_temp
        kwargs["model"] = T3CausalLM.from_pretrained_base_obj(
            base_lm,
            guidance_kwargs=self.guidance_kwargs,
            pooling=pooling,
            pool_temp=pool_temp,
            extraction_layer=extraction_layer,
            guidance_scale=guidance_scale,
            base_temp=base_temp
        )

        super().__init__(*args, **kwargs)

        # Make sure the right parameters are frozen/unfrozen
        assert not any([p.requires_grad for p in self.model.base_lm.parameters()])
        assert all([p.requires_grad for p in self.model.guidance_head.parameters()])

        logger.info("Initialized T3 trainer")

    def _prep_classifier_loss_from_tokens(self, model, inputs, split):
        outputs = model(
            input_ids=inputs[split]["input_ids"],
            attention_mask=inputs[split]["attention_mask"],
            labels=inputs[split]["labels"],
            classifier_only=True
        )

        shifted_labels = inputs[split]["labels"][:,1:].contiguous()
        classifier_labels = torch.full_like(shifted_labels, int(split == "retain"))
        classifier_labels[shifted_labels == IGNORE_INDEX] = IGNORE_INDEX  #(batch, seq_len-1)
        classifier_logits = outputs.classifier_logits.contiguous()  # (batch, seq_len-1)
        return classifier_logits.flatten(), classifier_labels.flatten()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise Exception(
                "\n[compute_loss] Unexpected call with return_outputs=True.\n"
                "Model inference with classifier adjustment is implemented "
                "only in prediction_step.\n"
                "Check your Trainer or evaluation code — this path should not "
                "be triggered during training."
            )

        retain_classifier_logits, retain_classifier_labels = self._prep_classifier_loss_from_tokens(
            model, inputs, "retain"
        )
        forget_classifier_logits, forget_classifier_labels = self._prep_classifier_loss_from_tokens(
            model, inputs, "forget"
        )


        # classifier_logits = torch.cat((retain_classifier_logits,forget_classifier_logits))
        # classifier_labels = torch.cat((retain_classifier_labels,forget_classifier_labels))

        # valid_mask = classifier_labels != -100
        # classifier_logits = classifier_logits[valid_mask]
        # classifier_labels = classifier_labels[valid_mask]

        # # Change from int to float
        # classifier_labels = classifier_labels.to(classifier_logits.dtype)
        # return F.binary_cross_entropy_with_logits(classifier_logits, classifier_labels)

        loss_sum = 0.0
        total_valid = 0

        for logits, labels in (
            (retain_classifier_logits, retain_classifier_labels),
            (forget_classifier_logits, forget_classifier_labels),
        ):
            valid = labels != -100
            if valid.any():
                y = labels[valid].to(logits.dtype)
                x = logits[valid]
                loss_sum = loss_sum + F.binary_cross_entropy_with_logits(x, y, reduction="sum")
                total_valid += valid.sum()

        return loss_sum / total_valid.clamp_min(1)
