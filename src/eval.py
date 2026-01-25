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
    # should be a DictConfig. Convert to container to ensure proper iteration
    from omegaconf import OmegaConf, open_dict
    # Convert to regular dict to ensure proper iteration
    eval_cfgs_dict = OmegaConf.to_container(eval_cfgs, resolve=False) if isinstance(eval_cfgs, DictConfig) else eval_cfgs
    # If it's a single config dict (has handler), wrap it
    if isinstance(eval_cfgs_dict, dict) and 'handler' in eval_cfgs_dict and len(eval_cfgs_dict) > 1:
        # It's a single config, wrap it in a dict with the eval name
        # Try to get the name from Hydra override or use default
        eval_name = 'trajectory_test'  # Default
        eval_cfgs_dict = {eval_name: eval_cfgs_dict}
    evaluators = get_evaluators(eval_cfgs_dict)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
