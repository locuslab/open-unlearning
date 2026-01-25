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
    # When using eval=trajectory_test, cfg.eval might be the config directly
    # or a dict. Check if it's a single config (has handler) vs dict of configs
    from omegaconf import OmegaConf, open_dict
    with open_dict(eval_cfgs):
        has_handler = eval_cfgs.get('handler') is not None
    # Check if it's a dict with multiple evaluators or a single config
    if has_handler and not (hasattr(eval_cfgs, 'keys') and len(list(eval_cfgs.keys())) > 1):
        # It's a direct config, wrap it in a dict
        # Try to infer the name from Hydra's override or use a default
        eval_name = 'trajectory_test'  # Default for our use case
        # Check if there's a trajectory_test key in the parent
        with open_dict(cfg):
            if hasattr(cfg.eval, 'trajectory_test'):
                eval_cfgs = {'trajectory_test': cfg.eval.trajectory_test}
            else:
                eval_cfgs = {eval_name: eval_cfgs}
    elif not hasattr(eval_cfgs, 'items'):
        # Convert to dict if needed
        eval_cfgs = OmegaConf.to_container(eval_cfgs, resolve=False) if isinstance(eval_cfgs, DictConfig) else eval_cfgs
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
