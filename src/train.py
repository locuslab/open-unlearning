import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything
from utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # Setup logging - use Hydra's output directory if available
    try:
        from hydra.core.global_hydra import GlobalHydra
        hydra_cfg = GlobalHydra.instance().hydra
        output_dir = hydra_cfg.runtime.output_dir
        log_file = f"{output_dir}/train.log"
    except:
        log_file = None
    setup_logging(log_file=log_file)
    
    logger.info("=" * 80)
    logger.info("Starting training/unlearning process")
    logger.info("=" * 80)
    logger.info(f"Task name: {cfg.get('task_name', 'N/A')}")
    logger.info(f"Mode: {cfg.get('mode', 'train')}")
    logger.info(f"Seed: {cfg.trainer.args.seed}")
    
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    
    logger.info(f"Loading model: {model_cfg.model_args.pretrained_model_name_or_path}")
    model, tokenizer = get_model(model_cfg)
    logger.info("Model loaded successfully")

    # Load Dataset
    logger.info("Loading datasets...")
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )
    train_size = len(data.get("train", [])) if data.get("train") is not None else 0
    eval_size = len(data.get("eval", [])) if data.get("eval") is not None else 0
    logger.info(f"Dataset loaded - Train: {train_size} samples, Eval: {eval_size} samples")

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    logger.info("Data collator loaded")

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")
    logger.info(f"Trainer: {trainer_cfg.handler}")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        logger.info("Loading evaluators...")
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )
        logger.info(f"Loaded {len(evaluators)} evaluator(s)")

    logger.info("Initializing trainer...")
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )
    logger.info(f"Output directory: {trainer_args.output_dir}")

    if trainer_args.do_train:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed")
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        logger.info(f"Model saved to {trainer_args.output_dir}")

    if trainer_args.do_eval:
        logger.info("Starting evaluation...")
        trainer.evaluate(metric_key_prefix="eval")
        logger.info("Evaluation completed")
    
    logger.info("=" * 80)
    logger.info("Process completed successfully")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
