import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from evals import get_evaluators
from model import get_model
from trainer.utils import seed_everything
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to evaluate
    """
    # Setup logging - use Hydra's output directory if available
    try:
        hydra_cfg = GlobalHydra.instance().hydra
        output_dir = hydra_cfg.runtime.output_dir
        log_file = f"{output_dir}/eval.log"
    except:
        log_file = None
    setup_logging(log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Starting evaluation process")
    logger.info("=" * 80)
    logger.info(f"Task name: {cfg.get('task_name', 'N/A')}")
    logger.info(f"Seed: {cfg.seed}")
    
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in eval config."
    
    logger.info(f"Loading model: {model_cfg.model_args.pretrained_model_name_or_path}")
    model, tokenizer = get_model(model_cfg)
    logger.info("Model loaded successfully")

    eval_cfgs = cfg.eval
    logger.info(f"Loading {len(eval_cfgs)} evaluator(s)...")
    evaluators = get_evaluators(eval_cfgs)
    
    for evaluator_name, evaluator in evaluators.items():
        logger.info(f"Running evaluation: {evaluator_name}")
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)
        logger.info(f"Completed evaluation: {evaluator_name}")
    
    logger.info("=" * 80)
    logger.info("Evaluation process completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
