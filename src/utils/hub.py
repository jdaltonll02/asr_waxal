"""Mirror a run's checkpoint directory to a public Hugging Face model repo.

Kaggle sessions end after 12 h and /kaggle/working is not carried into the next
session, so every Trainer save is mirrored to the Hub and pulled back on start.
That lets a rerun of the same notebook resume where the last session stopped.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import TrainerCallback

logger = logging.getLogger("waxal_net")


class HubMirror:
    def __init__(self, repo_id: str, local_dir: str):
        self.api = HfApi()
        self.repo_id = repo_id if "/" in repo_id else f"{self.api.whoami()['name']}/{repo_id}"
        self.local_dir = local_dir
        # Public: private storage is capped per account, public storage is not.
        self.api.create_repo(self.repo_id, private=False, exist_ok=True)

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
        self.squash(message)
        logger.info("Pushed %s to hf.co/%s (%s)", self.local_dir, self.repo_id, message)

    def squash(self, message: str) -> None:
        # Deleted checkpoints stay in git history and count toward the account's
        # storage; keep only the current files.
        try:
            self.api.super_squash_history(repo_id=self.repo_id, commit_message=message)
        except Exception as error:  # storage hygiene only; never fail training for it
            logger.warning("History squash failed: %s", error)

    def drop_intermediate_checkpoints(self) -> None:
        folders = {path.split("/")[0] for path in self.api.list_repo_files(self.repo_id) if path.startswith("checkpoint-")}
        for folder in sorted(folders):
            self.api.delete_folder(folder, repo_id=self.repo_id, commit_message=f"Remove {folder}")
        self.squash("best_model only")


class HubSyncCallback(TrainerCallback):
    def __init__(self, mirror: HubMirror):
        self.mirror = mirror

    def on_save(self, args, state, control, **kwargs):
        # Runs after the Trainer has written and rotated checkpoints.
        if state.is_world_process_zero:
            self.mirror.push(f"checkpoint at step {state.global_step}")


class ProgressCallback(TrainerCallback):
    """Log timed progress to run.log and, every few minutes, to progress.json on the Hub.

    Kaggle shows a job's logs only after it ends, so this is the live signal.
    """

    def __init__(self, mirror: HubMirror | None, every_seconds: float = 600):
        self.mirror = mirror
        self.every_seconds = every_seconds
        self.started = self.last_push = time.time()
        self.first_step: int | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        now = time.time()
        if self.first_step is None:  # time from the first logged step, excluding data loading and resume
            self.first_step, self.started = state.global_step, now
        steps = state.global_step - self.first_step
        progress = {"step": state.global_step, "max_steps": state.max_steps, "epoch": round(state.epoch or 0, 2),
                    "sec_per_step": round((now - self.started) / steps, 1) if steps else None,
                    "best_eval_loss": state.best_metric, **(logs or {}),
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now))}
        logger.info("progress %s", json.dumps(progress))
        if self.mirror and now - self.last_push >= self.every_seconds:
            self.last_push = now
            try:
                self.mirror.api.upload_file(path_or_fileobj=json.dumps(progress, indent=2).encode(), path_in_repo="progress.json",
                                            repo_id=self.mirror.repo_id, commit_message=f"progress: step {state.global_step}")
            except Exception as error:  # progress is best-effort; never stop training for it
                logger.warning("progress upload failed: %s", error)


def has_complete_model(path: str | Path) -> bool:
    return (Path(path) / "config.json").exists()
