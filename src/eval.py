import hydra
import logging
import os
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Enable verbose logging for HuggingFace
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
os.environ.setdefault("HF_HUB_VERBOSITY", "1")  # 1=info, 2=debug

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    logger.info("=== Starting evaluation ===")
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.get("template_args", None)
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    logger.info(f"Loading model: {model_cfg.get('model_args', {}).get('pretrained_model_name_or_path', 'unknown')}")
    model, tokenizer = get_model(model_cfg)
    logger.info("Model and tokenizer loaded successfully")

    eval_cfgs = cfg.eval
    # When using eval=trajectory_test, Hydra loads the config and cfg.eval
    # should be a DictConfig. Handle both dict of configs and single config
    from omegaconf import OmegaConf, open_dict
    # Check if it's a single config (has handler) vs dict of configs
    with open_dict(eval_cfgs):
        has_handler = eval_cfgs.get('handler') is not None
    # If it's a single config, wrap it in a dict
    if has_handler:
        eval_name = 'trajectory_test'  # Default
        eval_cfgs = {eval_name: eval_cfgs}
    # Ensure it's a DictConfig for get_evaluators
    if not isinstance(eval_cfgs, DictConfig):
        eval_cfgs = OmegaConf.create(eval_cfgs)
    logger.info("Getting evaluators...")
    evaluators = get_evaluators(eval_cfgs)
    logger.info(f"Found {len(evaluators)} evaluator(s): {list(evaluators.keys())}")
    for evaluator_name, evaluator in evaluators.items():
        logger.info(f"Running evaluator: {evaluator_name}")
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)
        logger.info(f"Evaluator {evaluator_name} completed")
    logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
