"""Build, push and collect the WAXAL-NET Kaggle jobs.

    python kaggle/jobs.py build                 # regenerate kaggle/kernels/*
    python kaggle/jobs.py push-token            # upload your HF write token as a private dataset
    python kaggle/jobs.py push-code             # upload src/, configs/, ... as the code dataset
    python kaggle/jobs.py push ft-acholi        # start every job whose slug contains a pattern
    python kaggle/jobs.py status [pattern ...]
    python kaggle/jobs.py progress [pattern ...]  # live step/speed/loss from each job's Hub repo
    python kaggle/jobs.py collect               # download job outputs, write results tables
    python kaggle/jobs.py queue ft-luganda ...  # keep Kaggle's 2 GPU slots busy until all finish

Each fine-tuning job is one (language, model) pair so it fits Kaggle's 12 h
session; rerunning a job resumes from checkpoints mirrored to the HF Hub.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KAGGLE_USER = "johndaltongibson"
CODE_DATASET = "waxal-asr-code"
# Private dataset holding the HF write token. Kaggle secrets are dropped from
# versions started with `kaggle kernels push`, so jobs read the token from here.
TOKEN_DATASET = "waxal-hf-token"
ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = ROOT / "kaggle" / "kernels"
RESULTS_DIR = ROOT / "results"

# Smallest training set first (paper Table 6), so early jobs validate the pipeline quickly.
LANGUAGES = ["acholi", "luganda", "soga", "masaaba", "nyankole", "akan", "lingala", "dagbani", "ewe", "shona",
             "ikposo", "dagaare", "malagasy", "fula", "tigrinya", "amharic", "oromo", "sidama", "wolaytta"]
FINETUNE_MODELS = ["whisper_tiny", "whisper_small", "mms_300m"]
# Paper §4.3: the six WAXAL languages that FLEURS also covers.
FLEURS_LANGUAGES = {"amharic", "fula", "lingala", "luganda", "oromo", "shona"}
ZERO_SHOT_JOBS = {
    "mms_1b": [LANGUAGES[:10], LANGUAGES[10:]],
    "omnilingual_1b": [LANGUAGES[:10], LANGUAGES[10:]],
    "whisper_large_v3": [LANGUAGES],  # 15 of 19 are N/A and finish instantly
}
TRAIN_HOURS = 10.5  # leaves ~1.5 h of the 12 h session for setup and evaluation

SETUP = '''import os, subprocess, sys, time
from pathlib import Path

SESSION_START = time.time()


def find_repo():
    for depth in range(1, 5):
        for hit in Path("/kaggle/input").glob("*/" * depth + "src/train.py"):
            return hit.parent.parent
    raise FileNotFoundError("Code not found: attach the waxal-asr-code dataset (Add Input).")


REPO = find_repo()
PY = "/tmp/waxal-venv/bin/python"
os.environ.update(HF_HOME="/tmp/hf_cache", HF_DATASETS_DISABLE_PROGRESS_BARS="1", WANDB_DISABLED="true",
                  PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  # One GPU: Trainer's DataParallel over both T4s made MMS-300M steps ~6.5x
                  # slower (kaggle/bench_mms.py). Accumulation doubles to keep batch 32.
                  CUDA_VISIBLE_DEVICES="0")
TOKEN_FILES = sorted(Path("/kaggle/input").glob("**/hf_token.txt"))
if TOKEN_FILES:
    os.environ["HF_TOKEN"] = TOKEN_FILES[0].read_text().strip()
else:
    print("No hf_token.txt input (waxal-hf-token dataset); checkpoints will not survive this session.")


LOG_DIR = Path("/kaggle/working/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run(*command, check=True):
    """Run a command from the repo root, streaming its output to the notebook and to
    /kaggle/working/logs (kept in the job output; the API's notebook log can be empty)."""
    name = "_".join(Path(str(part)).stem for part in command[1:4])
    log_path = LOG_DIR / f"{time.strftime('%H%M%S')}_{name}.log"
    process = subprocess.Popen(list(command), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    with open(log_path, "w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    if process.wait() and check:
        raise RuntimeError(f"{' '.join(map(str, command))} exited with {process.returncode}")
    return process.returncode


print("Code:", REPO)'''

FINETUNE_TRAIN = '''RUN = f"finetune_{LANGUAGE}_{MODEL}"
CONFIG = f"configs/experiment/{RUN}.yaml"
CHECKPOINT = f"/kaggle/working/checkpoints/{RUN}/best_model"
HUB = ["--hub-repo", f"waxal-net-{RUN}"] if os.environ.get("HF_TOKEN") else []
run(PY, "-m", "src.train", "--config", CONFIG, "--deadline", str(SESSION_START + TRAIN_HOURS * 3600), *HUB)'''

FINETUNE_EVALUATE = '''if Path(CHECKPOINT, "config.json").exists():
    run(PY, "-m", "src.evaluate", "--config", CONFIG, "--checkpoint", CHECKPOINT)
    if FLEURS:
        run(PY, "-m", "src.cross_domain_eval", "--config", f"configs/experiment/cross_domain_{LANGUAGE}_{MODEL}.yaml",
            "--checkpoint", CHECKPOINT)
else:
    print("Training stopped at the session deadline. Run this notebook again to resume from the last checkpoint.")'''

FINETUNE_SUMMARY = '''import json, shutil

for stage in ("finetune", "cross_domain"):
    path = Path(f"/kaggle/working/outputs/{stage}_{LANGUAGE}_{MODEL}/metrics.json")
    if path.exists():
        metrics = json.loads(path.read_text())
        print(f"{stage}: WER {metrics['wer_percent']:.1f}% | CER {metrics['cer_percent']:.1f}%")

# Keep the saved notebook output small. With HF_TOKEN every checkpoint and the
# final model are already on the Hub; without it only best_model is kept.
checkpoints = Path(f"/kaggle/working/checkpoints/{RUN}")
if os.environ.get("HF_TOKEN"):
    shutil.rmtree(checkpoints, ignore_errors=True)
else:
    for intermediate in checkpoints.glob("checkpoint-*"):
        shutil.rmtree(intermediate)'''

ZERO_SHOT_RUN = '''failed = {}
for language in LANGUAGES:
    if time.time() > SESSION_START + 11 * 3600:
        failed[language] = "skipped: session time"
        continue
    print(f"===== {language} =====", flush=True)
    if run(PY, "-m", "src.zero_shot_eval", "--config", f"configs/experiment/zero_shot_{language}_{MODEL}.yaml", check=False):
        failed[language] = "failed (see log above)"

import json
for language in LANGUAGES:
    path = Path(f"/kaggle/working/outputs/zero_shot_{language}_{MODEL}/metrics.json")
    metrics = json.loads(path.read_text()) if path.exists() else {}
    status = metrics.get("status") or (f"WER {metrics['wer_percent']:.1f}% | CER {metrics['cer_percent']:.1f}%" if metrics else failed.get(language))
    print(f"{language}: {status}")'''


# Whisper models on the Hub to re-score with chunked long audio (Acholi Tiny was
# trained before checkpoints were mirrored, so it has no Hub copy).
CHUNK30_RUNS = [("acholi", "whisper_small"), ("luganda", "whisper_small"), ("luganda", "whisper_tiny")]
CHUNK30_EVAL = '''import shutil

for language, model in RUNS:
    name = f"finetune_{language}_{model}"
    local = f"/kaggle/working/models/{name}"
    run(PY, "-c", "from huggingface_hub import HfApi, snapshot_download; "
        f"snapshot_download(HfApi().whoami()['name'] + '/waxal-net-{name}', allow_patterns='best_model/*', local_dir='{local}')")
    run(PY, "-m", "src.evaluate", "--config", f"configs/experiment/{name}.yaml", "--checkpoint", f"{local}/best_model",
        "--long-audio", "chunk30", "--output-dir", f"/kaggle/working/outputs/{name}_chunk30", check=False)
shutil.rmtree("/kaggle/working/models", ignore_errors=True)'''


def notebook(cells: list[tuple[str, str]]) -> dict:
    return {
        "cells": [{"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True),
                   **({"execution_count": None, "outputs": []} if kind == "code" else {})} for kind, source in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def write_kernel(slug: str, cells: list[tuple[str, str]]) -> None:
    directory = KERNELS_DIR / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "notebook.ipynb").write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {"id": f"{KAGGLE_USER}/{slug}", "title": slug, "code_file": "notebook.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [f"{KAGGLE_USER}/{CODE_DATASET}", f"{KAGGLE_USER}/{TOKEN_DATASET}"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def slug_of(*parts: str) -> str:
    return "waxal-" + "-".join(parts).replace("_", "-")


def build(_args) -> None:
    shutil.rmtree(KERNELS_DIR, ignore_errors=True)
    for language in LANGUAGES:
        for model in FINETUNE_MODELS:
            fleurs = language in FLEURS_LANGUAGES
            intro = (f"# WAXAL-NET fine-tune: {language} × {model}\n\n"
                     f"Trains on WAXAL {language}, evaluates on the WAXAL test split"
                     f"{' and 300 FLEURS test utterances' if fleurs else ''}.\n\n"
                     "Needs GPU T4 x2, internet, and the private `waxal-hf-token` dataset (an HF write token) so "
                     "checkpoints survive the 12 h session limit. If a run stops at the deadline, run it again to resume.")
            params = f'LANGUAGE = "{language}"\nMODEL = "{model}"\nFLEURS = {fleurs}\nTRAIN_HOURS = {TRAIN_HOURS}'
            write_kernel(slug_of("ft", language, model), [
                ("markdown", intro), ("code", params), ("code", SETUP),
                ("code", 'run("bash", f"{REPO}/kaggle/setup_env.sh")'),
                ("code", FINETUNE_TRAIN), ("code", FINETUNE_EVALUATE), ("code", FINETUNE_SUMMARY)])
    for model, groups in ZERO_SHOT_JOBS.items():
        for index, languages in enumerate(groups):
            suffix = [chr(ord("a") + index)] if len(groups) > 1 else []
            install = '"--omnilingual"' if model == "omnilingual_1b" else ""
            write_kernel(slug_of("zs", model, *suffix), [
                ("markdown", f"# WAXAL-NET zero-shot: {model}\n\nEvaluates on the WAXAL test split for: {', '.join(languages)}."),
                ("code", f'MODEL = "{model}"\nLANGUAGES = {languages!r}'), ("code", SETUP),
                ("code", f'run("bash", f"{{REPO}}/kaggle/setup_env.sh"{", " + install if install else ""})'),
                ("code", ZERO_SHOT_RUN)])
    write_kernel("waxal-bench-mms", [
        ("markdown", "# MMS-300M step-time benchmark\n\nTimes training steps on synthetic audio under several settings."),
        ("code", SETUP), ("code", 'run("bash", f"{REPO}/kaggle/setup_env.sh")'),
        ("code", 'run(PY, f"{REPO}/kaggle/bench_mms.py")')])
    write_kernel("waxal-eval-chunk30", [
        ("markdown", "# Whisper long-audio re-evaluation\n\nRe-scores fine-tuned Whisper models from the Hub with "
                     "clips over 30 s transcribed in 30 s chunks instead of truncated."),
        ("code", f"RUNS = {CHUNK30_RUNS!r}"), ("code", SETUP), ("code", 'run("bash", f"{REPO}/kaggle/setup_env.sh")'),
        ("code", CHUNK30_EVAL)])
    print(f"Wrote {len(list(KERNELS_DIR.iterdir()))} kernels to {KERNELS_DIR.relative_to(ROOT)}")


def kaggle(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$ kaggle", " ".join(arguments), flush=True)
    # UTF-8 output: on Windows the CLI otherwise crashes printing non-ASCII file names.
    return subprocess.run(["kaggle", *arguments], check=check, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def push_code(args) -> None:
    with tempfile.TemporaryDirectory() as staging:
        code = Path(staging) / "asr_waxal"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "kernels")
        for folder in ("src", "configs", "kaggle"):
            shutil.copytree(ROOT / folder, code / folder, ignore=ignore)
        for name in ("requirements.txt", "requirements-omnilingual.txt", "README.md"):
            shutil.copy2(ROOT / name, code / name)
        metadata = {"title": CODE_DATASET, "id": f"{KAGGLE_USER}/{CODE_DATASET}", "licenses": [{"name": "CC0-1.0"}]}
        (Path(staging) / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # --dir-mode: the CLI silently skips sub-folders by default.
        if kaggle("datasets", "status", f"{KAGGLE_USER}/{CODE_DATASET}", check=False).returncode == 0:
            kaggle("datasets", "version", "-p", staging, "-m", args.message, "--dir-mode", "zip")
        else:
            kaggle("datasets", "create", "-p", staging, "--dir-mode", "zip")


def push_token(_args) -> None:
    from huggingface_hub import get_token, whoami
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("No HF token: run `hf auth login --force` (write token) or set HF_TOKEN.")
    try:
        identity = whoami(token)
    except Exception as error:
        sys.exit(f"HF token rejected ({error}). Run `hf auth login --force` with a write token.")
    role = identity.get("auth", {}).get("accessToken", {}).get("role")
    if role not in ("write", "fineGrained"):
        sys.exit(f"Token role is {role!r}; jobs need a write token to create repos and push checkpoints.")
    with tempfile.TemporaryDirectory() as staging:
        (Path(staging) / "hf_token.txt").write_text(token, encoding="utf-8")
        metadata = {"title": TOKEN_DATASET, "id": f"{KAGGLE_USER}/{TOKEN_DATASET}", "licenses": [{"name": "CC0-1.0"}]}
        (Path(staging) / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Datasets are private unless --public is passed; never pass it here.
        if kaggle("datasets", "status", f"{KAGGLE_USER}/{TOKEN_DATASET}", check=False).returncode == 0:
            kaggle("datasets", "version", "-p", staging, "-m", "Rotate token")
        else:
            kaggle("datasets", "create", "-p", staging)
    print(f"Uploaded token for HF user {identity['name']} to private dataset {KAGGLE_USER}/{TOKEN_DATASET}.")


def selected(patterns: list[str]) -> list[Path]:
    kernels = sorted(KERNELS_DIR.iterdir(), key=lambda path: kernel_order(path.name))
    return [path for path in kernels if not patterns or any(pattern in path.name for pattern in patterns)]


def kernel_order(slug: str) -> tuple:
    for index, language in enumerate(LANGUAGES):
        if f"-{language}-" in slug:
            return (0, index, slug)
    return (1, 0, slug)


def push(args) -> None:
    if not args.patterns and not args.all:
        sys.exit("Name slug patterns to push (e.g. ft-acholi), or pass --all.")
    for path in selected(args.patterns):
        kaggle("kernels", "push", "-p", str(path), check=False)


def status(args) -> None:
    for path in selected(args.patterns):
        kaggle("kernels", "status", f"{KAGGLE_USER}/{path.name}", check=False)


def progress(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    owner = api.whoami()["name"]
    for path in selected(args.patterns):
        if not path.name.startswith("waxal-ft-"):
            continue
        language, model = path.name.removeprefix("waxal-ft-").split("-", 1)
        repo = f"{owner}/waxal-net-finetune_{language}_{model.replace('-', '_')}"
        try:
            state = json.loads(Path(hf_hub_download(repo, "progress.json", force_download=True)).read_text())
            print(f"{path.name}: step {state['step']} | {state.get('sec_per_step')} s/step | "
                  f"loss {state.get('loss', state.get('eval_loss'))} | best eval {state.get('best_eval_loss')} | {state['updated']}")
        except Exception as error:
            print(f"{path.name}: no progress yet ({type(error).__name__})")


def collect(args) -> None:
    outputs = RESULTS_DIR / "outputs"
    for path in selected(args.patterns):
        target = RESULTS_DIR / "kernels" / path.name
        target.mkdir(parents=True, exist_ok=True)
        # The CLI can exit non-zero after downloading everything, so use whatever arrived.
        if kaggle("kernels", "output", f"{KAGGLE_USER}/{path.name}", "-p", str(target), check=False).returncode:
            print(f"warning: download of {path.name} reported an error; using the files that arrived")
        for metrics in (found for found in target.rglob("metrics.json") if found.parent.parent.name == "outputs"):
            destination = outputs / metrics.parent.name
            shutil.copytree(metrics.parent, destination, dirs_exist_ok=True)
    subprocess.run([sys.executable, "-m", "src.report_results", "--outputs-dir", str(outputs),
                    "--output", str(RESULTS_DIR / "waxal_results.md")], cwd=ROOT, check=True)


MAX_SESSIONS = 2  # Kaggle: "Maximum batch GPU session count of 2"
MAX_SESSIONS_PER_JOB = 4
QUEUE_STATE = RESULTS_DIR / "queue_state.json"


def kernel_state(slug: str) -> str:
    result = subprocess.run(["kaggle", "kernels", "status", f"{KAGGLE_USER}/{slug}"], capture_output=True, text=True,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    text = result.stdout + result.stderr
    for state in ("CANCEL", "COMPLETE", "ERROR", "RUNNING", "QUEUED"):
        if state in text:
            return state
    return "UNKNOWN"


def job_outcome(slug: str) -> str:
    """'done', 'resume' (stopped at the session deadline) or 'failed', from the job's saved output."""
    target = RESULTS_DIR / "kernels" / slug
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    kaggle("kernels", "output", f"{KAGGLE_USER}/{slug}", "-p", str(target), check=False)
    if not slug.startswith("waxal-ft-"):
        return "done" if list(target.rglob("metrics.json")) else "failed"
    statuses = list(target.rglob("training_status.json"))
    if statuses and not json.loads(statuses[0].read_text(encoding="utf-8")).get("complete"):
        return "resume"
    finetune_metrics = [path for path in target.rglob("metrics.json") if path.parent.name.startswith("finetune_")]
    return "done" if finetune_metrics else "failed"


def push_one(slug: str) -> bool:
    result = subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNELS_DIR / slug)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    text = (result.stdout + result.stderr).strip()
    print(f"  push {slug}: {text.splitlines()[-1] if text else result.returncode}", flush=True)
    return "successfully pushed" in text


def queue(args) -> None:
    """Run jobs two at a time; resume deadline stops; stop retrying failures. Safe to restart."""
    import time
    state = json.loads(QUEUE_STATE.read_text(encoding="utf-8")) if QUEUE_STATE.exists() else {}
    # Run in the order the patterns were given, then smallest language first.
    def priority(path):
        return (next(index for index, pattern in enumerate(args.patterns) if pattern in path.name), kernel_order(path.name))
    names = [path.name for path in sorted(selected(args.patterns), key=priority)]
    for name in names:
        state.setdefault(name, {"status": "pending", "sessions": 0})

    def save():
        QUEUE_STATE.parent.mkdir(parents=True, exist_ok=True)
        QUEUE_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")

    while True:
        for name in names:
            job = state[name]
            if job["status"] != "running":
                continue
            kernel = kernel_state(name)
            if kernel in ("RUNNING", "QUEUED", "UNKNOWN"):
                continue
            outcome = job_outcome(name) if kernel == "COMPLETE" else "failed"
            if outcome == "resume" and job["sessions"] >= MAX_SESSIONS_PER_JOB:
                outcome = "failed"
            job["status"] = {"done": "done", "resume": "pending", "failed": "failed"}[outcome]
            print(f"{time.strftime('%H:%M')} {name}: {kernel} -> {outcome} (sessions: {job['sessions']})", flush=True)
            save()
        running = sum(state[name]["status"] == "running" for name in names)
        for name in names:
            if running >= MAX_SESSIONS:
                break
            if state[name]["status"] == "pending" and push_one(name):
                state[name]["status"] = "running"
                state[name]["sessions"] += 1
                running += 1
                save()
        open_jobs = [name for name in names if state[name]["status"] in ("pending", "running")]
        if not open_jobs:
            break
        time.sleep(args.poll)
    summary = {status: [name for name in names if state[name]["status"] == status] for status in ("done", "failed")}
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build").set_defaults(handler=build)
    commands.add_parser("push-token").set_defaults(handler=push_token)
    code = commands.add_parser("push-code")
    code.add_argument("-m", "--message", default="Update code")
    code.set_defaults(handler=push_code)
    pusher = commands.add_parser("push")
    pusher.add_argument("patterns", nargs="*")
    pusher.add_argument("--all", action="store_true")
    pusher.set_defaults(handler=push)
    queuer = commands.add_parser("queue")
    queuer.add_argument("patterns", nargs="+")
    queuer.add_argument("--poll", type=int, default=300, help="seconds between status checks")
    queuer.set_defaults(handler=queue)
    for name, handler in (("status", status), ("progress", progress), ("collect", collect)):
        command = commands.add_parser(name)
        command.add_argument("patterns", nargs="*")
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
