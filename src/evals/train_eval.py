"""
Custom evaluator for evaluating on training datasets (forget/retain).
This evaluator can accept datasets directly instead of loading from config.
"""
import logging
from evals.base import Evaluator
from typing import Optional, Union
from torch.utils.data import Dataset

logger = logging.getLogger("evaluator")


class TrainDatasetEvaluator(Evaluator):
    """
    Evaluator that can evaluate on datasets passed directly from the trainer.
    This is useful for evaluating on the forget/retain datasets used during training.
    """
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("TrainDataset", eval_cfg, **kwargs)
        self.dataset = None  # Can be set later via set_dataset method
    
    def set_dataset(self, dataset: Union[Dataset, list]):
        """Set the dataset to evaluate on."""
        self.dataset = dataset
    
    def evaluate(self, dataset=None, model=None, output_dir=None, overwrite=None, **kwargs):
        """
        Evaluate on the provided dataset or the dataset set via set_dataset.
        
        Args:
            dataset: Optional dataset to evaluate on. If provided, overrides self.dataset
            model: Model to evaluate  
            output_dir: Output directory for results
            overwrite: Whether to overwrite existing results
            **kwargs: Additional arguments (tokenizer, template_args, collators, etc.)
        """
        # Use provided dataset or self.dataset
        eval_dataset = dataset if dataset is not None else self.dataset
        
        if eval_dataset is None:
            raise ValueError("No dataset provided. Either pass dataset to evaluate() or call set_dataset() first.")
        
        # Set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        if model is not None:
            model = self.prepare_model(model)
        else:
            model = kwargs.get("model")
            if model is not None:
                model = self.prepare_model(model)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )
        
        # Get collators - either from kwargs or load from config
        collators = kwargs.get("collators", None)
        if collators is None:
            # Load collators from the first metric's config
            # The metrics have collator configs in their defaults
            first_metric_name = list(self.metrics.keys())[0] if self.metrics else None
            if first_metric_name:
                metric_cfg = self.eval_cfg.metrics.get(first_metric_name, {})
                # Check if metric has collator config (from defaults)
                if hasattr(metric_cfg, "collators"):
                    from data import get_collators
                    collators = get_collators(metric_cfg.collators, tokenizer=kwargs.get("tokenizer"))
                elif "collators" in metric_cfg:
                    from data import get_collators
                    collators = get_collators(metric_cfg["collators"], tokenizer=kwargs.get("tokenizer"))
            
            # If still no collators, try to use DataCollatorForSupervisedDatasetwithIndex as default
            if collators is None:
                from data.collators import DataCollatorForSupervisedDatasetwithIndex
                collators = DataCollatorForSupervisedDatasetwithIndex(tokenizer=kwargs.get("tokenizer"))
        
        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info(f"Skipping {metric_name}, already evaluated.")
                if "agg_value" in logs[metric_name]:
                    logger.info(
                        f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                    )
                self.save_logs(self.summarize(logs), summary_file_path)
                continue
            _ = logs.pop(metric_name, None)  # overwriting existing evals if present
            
            # Prepare kwargs for metric evaluation
            # Metrics expect: data, collators, batch_size, tokenizer, template_args
            # Note: model is passed as positional argument, not in kwargs
            metrics_args = self.eval_cfg.metrics.get(metric_name, {})
            metric_kwargs = {
                "tokenizer": kwargs.get("tokenizer", None),
                "template_args": kwargs.get("template_args", None),
                "data": eval_dataset,  # Pass the dataset directly
                "collators": collators,  # Pass collators
                "batch_size": metrics_args.get("batch_size", 32),
            }
            # Update with any additional metric-specific args
            metric_kwargs.update({k: v for k, v in metrics_args.items() 
                                 if k not in ["batch_size", "datasets", "collators"]})
            
            # Call prepare_kwargs_evaluate_metric if the metric needs it
            # But since we're passing data directly, we might not need it
            result = metric_fn._metric_fn(model, **metric_kwargs)
            
            if "agg_value" in result:
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            logs[metric_name] = result
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        return self.summarize(logs)
