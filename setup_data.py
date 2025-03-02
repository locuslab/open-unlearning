from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="open-unlearning/eval",
    allow_patterns="*.json",
    repo_type="dataset",
    local_dir="saves/eval",
)
