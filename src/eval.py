import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.get("template_args", None)
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

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
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
