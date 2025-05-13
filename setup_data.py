import argparse
import os
import subprocess
from huggingface_hub import snapshot_download


def download_eval_data():
    snapshot_download(
        repo_id="open-unlearning/eval",
        allow_patterns="*.json",
        repo_type="dataset",
        local_dir="saves/eval",
    )


def download_idk_data():
    snapshot_download(
        repo_id="open-unlearning/idk",
        allow_patterns="*.jsonl",
        repo_type="dataset",
        local_dir="data",
    )


def download_wmdp():
    url = "https://cais-wmdp.s3.us-west-1.amazonaws.com/wmdp-corpora.zip"
    dest_dir = "data/wmdp"
    zip_path = os.path.join(dest_dir, "wmdp-corpora.zip")

    os.makedirs(dest_dir, exist_ok=True)
    subprocess.run(["wget", url, "-O", zip_path], check=True)
    subprocess.run(["unzip", "-P", "wmdpcorpora", zip_path, "-d", dest_dir], check=True)


def main():
    parser = argparse.ArgumentParser(description="Download and setup evaluation data.")
    parser.add_argument(
        "--eval_logs",
        action="store_true",
        help="Downloads TOFU, MUSE  - retain and finetuned models eval logs and saves them in saves/eval",
    )
    parser.add_argument(
        "--idk",
        action="store_true",
        help="Download idk dataset from HF hub and stores it data/idk.jsonl",
    )
    parser.add_argument(
        "--wmdp",
        action="store_true",
        help="Download and unzip WMDP dataset into data/wmdp",
    )

    args = parser.parse_args()

    if args.eval_logs:
        download_eval_data()
    if args.idk:
        download_idk_data()
    if args.wmdp:
        download_wmdp()


if __name__ == "__main__":
    main()
