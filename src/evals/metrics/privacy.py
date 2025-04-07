import numpy as np
from scipy.stats import ks_2samp
from evals.metrics.base import unlearning_metric, logger


@unlearning_metric(name="ks_test")
def ks_test(model, **kwargs):
    """Compare two forget and retain model distributions with a 2-sample KS-test and report the p-value.
    Used in the TOFU benchmark as forget_quality when computed over the truth_ratio statistic."""
    forget_tr_stats = np.array(
        [
            evals["score"]
            for evals in kwargs["pre_compute"]["forget"]["value_by_index"].values()
        ]
    )
    reference_logs = kwargs.get("reference_logs", None)
    if reference_logs:
        reference_logs = reference_logs["retain_model_logs"]
        retain_tr_stats = np.array(
            [
                evals["score"]
                for evals in reference_logs["retain"]["value_by_index"].values()
            ]
        )
        fq = ks_2samp(forget_tr_stats, retain_tr_stats)
        pvalue = fq.pvalue
    else:
        logger.warning(
            "retain_model_logs not provided in reference_logs, setting forget_quality to None"
        )
        pvalue = None
    return {"agg_value": pvalue}


@unlearning_metric(name="privleak")
def privleak(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic.
    To be used for MIA AUC scores in ensuring consistency and reproducibility of the MUSE benchmark.
    This function is similar to the rel_diff function below, but due to the MUSE benchmark reporting AUC
    scores as (1-x) when the more conventional way is x, we do adjustments here to our MIA AUC scores.
    calculations in the reverse way,"""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    score = 1 - score
    ref = 1 - ref
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}


@unlearning_metric(name="rel_diff")
def rel_diff(model, **kwargs):
    """Compare two forget and retain model scores using a relative comparison of a single statistic."""
    score = kwargs["pre_compute"]["forget"]["agg_value"]
    try:
        ref = kwargs["reference_logs"]["retain_model_logs"]["retain"]["agg_value"]
    except Exception as _:
        logger.warning(
            f"retain_model_logs evals not provided for privleak, using default retain auc of {kwargs['ref_value']}"
        )
        ref = kwargs["ref_value"]
    return {"agg_value": (score - ref) / (ref + 1e-10) * 100}
