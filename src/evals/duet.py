from evals.base import Evaluator


class DUETEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("DUET", eval_cfg, **kwargs)
