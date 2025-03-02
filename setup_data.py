from huggingface_hub import snapshot_download

# Setup retain model metrics
snapshot_download(
    repo_id="open-unlearning/eval",
    allow_patterns="*.json",
    repo_type="dataset",
    local_dir="saves/eval",
)

# Setup data
snapshot_download(
    repo_id="open-unlearning/idk",
    allow_patterns="*.jsonl",
    repo_type="dataset",
    local_dir="data",
)