"""Download required models and datasets from Hugging Face and GitHub.
No need to run this on ECE cluster nodes.
"""
import os
import subprocess

import configs
from huggingface_hub import snapshot_download

ALGORITHMS_REPO_URL = "https://github.com/TheAlgorithms/Python.git"


def clone_or_pull(repo_url: str, local_path: str):
    """Clone a GitHub repo, or pull latest if it already exists."""
    if os.path.exists(os.path.join(local_path, ".git")):
        print(f"Repo already exists at '{local_path}', pulling latest")
        subprocess.run(["git", "-C", local_path, "pull"], check=True)
    else:
        os.makedirs(local_path, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, local_path], check=True)


def main():
    """Download all required models and datasets."""
    cfg = configs.Config()

    print(f"Downloading model {cfg.model_repo_id} - {cfg.model_name}")
    snapshot_download(
        repo_id=cfg.model_repo_id,
        local_dir=cfg.model_name,
    )

    clone_or_pull(ALGORITHMS_REPO_URL, cfg.data_dir)
    print("Done.")


if __name__ == "__main__":
    main()