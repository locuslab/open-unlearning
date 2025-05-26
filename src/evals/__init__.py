from typing import Dict, Any
from omegaconf import DictConfig
from evals.tofu import TOFUEvaluator
from evals.muse import MUSEEvaluator
from evals.lm_eval import LMEvalEvaluator

EVALUATOR_REGISTRY: Dict[str, Any] = {}


def _register_evaluator(evaluator_class):
    EVALUATOR_REGISTRY[evaluator_class.__name__] = evaluator_class


def get_evaluator(name: str, eval_cfg: DictConfig, **kwargs):
    evaluator_handler_name = eval_cfg.get("handler")
    assert evaluator_handler_name is not None, ValueError(f"{name} handler not set")
    eval_handler = EVALUATOR_REGISTRY.get(evaluator_handler_name)
    if eval_handler is None:
        raise NotImplementedError(
            f"{evaluator_handler_name} not implemented or not registered"
        )
    return eval_handler(eval_cfg, **kwargs)


def get_evaluators(eval_cfgs: DictConfig, **kwargs):
    evaluators = {}
    for eval_name, eval_cfg in eval_cfgs.items():
        evaluators[eval_name] = get_evaluator(eval_name, eval_cfg, **kwargs)
    return evaluators


# Register Your benchmark evaluators
_register_evaluator(TOFUEvaluator)
_register_evaluator(MUSEEvaluator)
_register_evaluator(LMEvalEvaluator)
