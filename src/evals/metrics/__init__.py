from typing import Dict
from omegaconf import DictConfig
from evals.metrics.base import UnlearningMetric
from evals.metrics.memorization import (
    probability,
    probability_w_options,
    rouge,
    truth_ratio,
    extraction_strength,
    exact_memorization,
)
from evals.metrics.privacy import ks_test, privleak, rel_diff
from evals.metrics.mia import (
    mia_loss,
    mia_min_k,
    mia_min_k_plus_plus,
    mia_gradnorm,
    mia_zlib,
    mia_reference,
)
from evals.metrics.utility import (
    hm_aggregate,
    classifier_prob,
)

METRICS_REGISTRY: Dict[str, UnlearningMetric] = {}


def _register_metric(metric):
    METRICS_REGISTRY[metric.name] = metric


def _get_single_metric(name: str, metric_cfg, **kwargs):
    metric_handler_name = metric_cfg.get("handler")
    assert metric_handler_name is not None, ValueError(f"{name} handler not set")
    metric = METRICS_REGISTRY.get(metric_handler_name)
    if metric is None:
        raise NotImplementedError(
            f"{metric_handler_name} not implemented or not registered"
        )
    pre_compute_cfg = metric_cfg.get("pre_compute", {})
    pre_compute_metrics = get_metrics(pre_compute_cfg, **kwargs)
    metric.set_pre_compute_metrics(pre_compute_metrics)
    return metric


def get_metrics(metric_cfgs: DictConfig, **kwargs):
    metrics = {}
    for metric_name, metric_cfg in metric_cfgs.items():
        metrics[metric_name] = _get_single_metric(metric_name, metric_cfg, **kwargs)
    return metrics


# Register metrics here
_register_metric(probability)
_register_metric(probability_w_options)
_register_metric(rouge)
_register_metric(truth_ratio)
_register_metric(ks_test)
_register_metric(hm_aggregate)
_register_metric(privleak)
_register_metric(rel_diff)
_register_metric(exact_memorization)
_register_metric(extraction_strength)

# Register MIA metrics
_register_metric(mia_loss)
_register_metric(mia_min_k)
_register_metric(mia_min_k_plus_plus)
_register_metric(mia_gradnorm)
_register_metric(mia_zlib)
_register_metric(mia_reference)

# Register Utility metrics
_register_metric(classifier_prob)
