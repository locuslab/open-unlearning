from __future__ import annotations

import subprocess
import sys

import modal

sys.path.insert(0, "/workspace/scripts")

from modal_project_setup import (  # noqa: E402
    WORKDIR,
    build_project_image,
    hf_cache,
    results,
    runtime_env,
)

APP_NAME = "open-unlearning-tofu-eval-hf-rmu-scan-llama32-1b"

app = modal.App(APP_NAME)
image = build_project_image()

RETAIN90_LOGS = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"

# Rationale for selection:
# Current best: layer5_scoeff100_lr1e-5 (QAP=0.426, utility≈0.55, harmonic=0.691)
# Tested and collapsed: layer10_scoeff100_lr{1e-5,2e-5} (either didn't forget or lost utility)
# Entirely untested: scoeff10 across all layers, layer15 entirely, lr5e-5 entirely
#
# (label, hf_model_id, output_dir)
HF_MODELS = [
    # --- Extend best-performing config (layer5 + scoeff100) to higher lr ---
    # Layer5 is stable at scoeff100; higher lr should push QAP lower without collapse.
    (
        "rmu_l5_s100_lr2e5",
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer5_scoeff100_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer5_scoeff100_epoch10_hf",
    ),
    (
        "rmu_l5_s100_lr5e5",
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff100_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff100_epoch10_hf",
    ),
    # --- Intermediate scoeff10 on layer5 (untested region) ---
    # High lr compensates for lower steering coefficient; tests whether scoeff matters less than lr.
    (
        "rmu_l5_s10_lr5e5",
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer5_scoeff10_epoch10_hf",
    ),
    # --- scoeff10 on layer10 (most downloaded untested checkpoint, 37 downloads) ---
    # scoeff10 avoids the utility collapse seen with scoeff100+layer10; highest lr to drive forgetting.
    (
        "rmu_l10_s10_lr5e5",
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr5e-05_layer10_scoeff10_epoch10_hf",
    ),
    # --- layer15 (entirely unexplored; deeper modification may localize unlearning better) ---
    (
        "rmu_l15_s100_lr1e5",
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer15_scoeff100_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr1e-05_layer15_scoeff100_epoch10_hf",
    ),
    (
        "rmu_l15_s100_lr2e5",
        "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff100_epoch10",
        "saves/eval/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_RMU_lr2e-05_layer15_scoeff100_epoch10_hf",
    ),
]


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{WORKDIR}/saves": results,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_eval(model_id: str, output_dir: str) -> None:
    subprocess.run(["python", "setup_data.py", "--eval"], check=True, cwd=WORKDIR, env=runtime_env())

    subprocess.run(
        [
            "python",
            "src/eval.py",
            "experiment=eval/tofu/default.yaml",
            "forget_split=forget10",
            "holdout_split=holdout10",
            "model=Llama-3.2-1B-Instruct",
            "task_name=tofu_Llama-3.2-1B-Instruct_hf_rmu_scan",
            f"model.model_args.pretrained_model_name_or_path={model_id}",
            f"paths.output_dir={output_dir}",
            f"retain_logs_path={RETAIN90_LOGS}",
        ],
        check=True,
        cwd=WORKDIR,
        env=runtime_env(),
    )

    results.commit()
    hf_cache.commit()


@app.local_entrypoint()
def main(run_only: str = "") -> None:
    wanted = {x.strip() for x in run_only.split(",") if x.strip()}

    for label, model_id, output_dir in HF_MODELS:
        if wanted and label not in wanted:
            continue
        run_eval.spawn(model_id=model_id, output_dir=output_dir)
