"""Mirror a run's checkpoint directory to a private Hugging Face model repo.

Kaggle sessions end after 12 h and /kaggle/working is not carried into the next
session, so every Trainer save is mirrored to the Hub and pulled back on start.
That lets a rerun of the same notebook resume where the last session stopped.
"""
from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import TrainerCallback

logger = logging.getLogger("waxal_net")


class HubMirror:
    def __init__(self, repo_id: str, local_dir: str):
        self.api = HfApi()
        self.repo_id = repo_id if "/" in repo_id else f"{self.api.whoami()['name']}/{repo_id}"
        self.local_dir = local_dir
        self.api.create_repo(self.repo_id, private=True, exist_ok=True)

    def pull(self) -> None:
        if not self.api.list_repo_files(self.repo_id):
            return
        snapshot_download(self.repo_id, local_dir=self.local_dir)
        logger.info("Restored %s from hf.co/%s", self.local_dir, self.repo_id)

    def push(self, message: str) -> None:
        # Remote checkpoint-* folders not present locally (rotated out by
        # save_total_limit) are deleted in the same commit; uploaded files are kept.
        self.api.upload_folder(repo_id=self.repo_id, folder_path=self.local_dir, commit_message=message,
                               delete_patterns=["checkpoint-*"])
        logger.info("Pushed %s to hf.co/%s (%s)", self.local_dir, self.repo_id, message)

    def drop_intermediate_checkpoints(self) -> None:
        folders = {path.split("/")[0] for path in self.api.list_repo_files(self.repo_id) if path.startswith("checkpoint-")}
        for folder in sorted(folders):
            self.api.delete_folder(folder, repo_id=self.repo_id, commit_message=f"Remove {folder}")


class HubSyncCallback(TrainerCallback):
    def __init__(self, mirror: HubMirror):
        self.mirror = mirror

    def on_save(self, args, state, control, **kwargs):
        # Runs after the Trainer has written and rotated checkpoints.
        if state.is_world_process_zero:
            self.mirror.push(f"checkpoint at step {state.global_step}")


def has_complete_model(path: str | Path) -> bool:
    return (Path(path) / "config.json").exists()
