import logging
from omegaconf import OmegaConf

from lm_eval.models.hf_vlms import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import simple_evaluate

from evals.base import Evaluator


logger = logging.getLogger("evaluator")


class LMEvalEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        self.name = "LMEval"
        self.eval_cfg = eval_cfg
        self.tasks = OmegaConf.to_container(
            self.eval_cfg.tasks, resolve=True, throw_on_missing=True
        )
        self.task_manager = TaskManager()
        self.simple_evaluate_args = dict(kwargs.get("simple_evaluate_args", {}))

    def prepare_model(self, model, **kwargs):
        """Prepare model for evaluation"""
        model.eval()
        return HFLM(model)

    def summarize(self, eval_results: dict, task_name: str) -> dict:
        """
        Summarize evaluation metrics from lm_eval.simple_evaluate.
        - If task_name is a group, return only aggregated group-level metrics.
        - If it's a single task, return per-task metrics from 'results'.
        - Always exclude 'alias' entries and strip ',none' suffixes.
        """
        summary = {}

        def clean_metric_key(prefix: str, metric_name: str) -> str | None:
            if metric_name == "alias":
                return None
            base = metric_name.split(",", 1)[0].strip()
            return f"{prefix}/{base}"

        # Check if task is a group (e.g., 'mmlu')
        if task_name in self.task_manager.all_groups:
            group_metrics = eval_results.get("groups", {}).get(task_name, {})
            for metric_name, value in group_metrics.items():
                key = clean_metric_key(task_name, metric_name)
                if key is None:
                    continue
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    summary[key] = value
        else:
            task_metrics = eval_results.get("results", {}).get(task_name, {})
            for metric_name, value in task_metrics.items():
                key = clean_metric_key(task_name, metric_name)
                if key is None:
                    continue
                try:
                    summary[key] = float(value)
                except (TypeError, ValueError):
                    summary[key] = value

        return summary

    def get_task_name(self, task):
        if isinstance(task, str):
            return task
        elif isinstance(task, dict):
            if "task" in task:
                return task.get("task")
        raise ValueError(f"Invalid task format: {task}")

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        kwargs = {"tokenizer": kwargs.get("tokenizer", None)}
        model = self.prepare_model(model, **kwargs)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}
        summary = self.load_logs_from_file(summary_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )

        for task in self.tasks:
            task_name = self.get_task_name(task)
            if not overwrite and task_name in logs and logs[task_name]:
                logger.info(f"Skipping {task_name}, already evaluated.")
                continue
            _ = logs.pop(task_name, None)  # overwriting existing evals if present
            results = simple_evaluate(
                model=model,
                tasks=[task],
                task_manager=self.task_manager,
                **self.simple_evaluate_args,
            )
            logs.update({task_name: results["samples"]})
            summary.update(self.summarize(results, task_name))
            self.save_logs(logs, logs_file_path)
            self.save_logs(summary, summary_file_path)
        return summary
