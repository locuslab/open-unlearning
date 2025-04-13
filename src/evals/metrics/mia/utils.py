from evals.metrics.mia.all_attacks import AllAttacks
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.reference import ReferenceAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack

from sklearn.metrics import roc_auc_score


import numpy as np


def get_attacker(attack: str):
    mapping = {
        AllAttacks.LOSS: LOSSAttack,
        AllAttacks.REFERENCE_BASED: ReferenceAttack,
        AllAttacks.ZLIB: ZLIBAttack,
        AllAttacks.MIN_K: MinKProbAttack,
        AllAttacks.MIN_K_PLUS_PLUS: MinKPlusPlusAttack,
        AllAttacks.GRADNORM: GradNormAttack,
    }
    attack_cls = mapping.get(attack, None)
    if attack_cls is None:
        raise ValueError(f"Attack {attack} not found")
    return attack_cls


def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    """
    Compute the MIA AUC and accuracy.

    Parameters:
      - attack_cls: the attack class to use.
      - model: the target model.
      - data: a dict with keys "forget" and "holdout".
      - collator: data collator.
      - batch_size: batch size.
      - kwargs: additional optional parameters (e.g. k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs, including "acc" and "auc".

    Note on convention: auc is 1 when the forget data is much more likely than the holdout data
    """
    # Build attack arguments from common parameters and any extras.
    attack_args = {
        "model": model,
        "collator": collator,
        "batch_size": batch_size,
    }
    attack_args.update(kwargs)

    output = {
        "forget": attack_cls(data=data["forget"], **attack_args).attack(),
        "holdout": attack_cls(data=data["holdout"], **attack_args).attack(),
    }
    forget_scores = [
        elem["score"] for elem in output["forget"]["value_by_index"].values()
    ]
    holdout_scores = [
        elem["score"] for elem in output["holdout"]["value_by_index"].values()
    ]
    scores = np.array(forget_scores + holdout_scores)
    labels = np.array(
        [0] * len(forget_scores) + [1] * len(holdout_scores)
    )  # see note above
    auc_value = roc_auc_score(labels, scores)
    output["auc"], output["agg_value"] = auc_value, auc_value
    return output
