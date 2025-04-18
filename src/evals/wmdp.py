from evals.base import Evaluator


class WMDPEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("WMDP", eval_cfg, **kwargs)
