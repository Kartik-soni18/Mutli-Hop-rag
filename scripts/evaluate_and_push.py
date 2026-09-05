"""Local orchestration; no API calls happen merely by importing this module."""

import json
import os
import secrets
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.dashboard import update_dashboard  # noqa: E402


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def clean_at(commit):
    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain"):
        raise RuntimeError(
            "Repository changed during evaluation. Results are preserved in evaluation/runs; no files overwritten or pushed."
        )


def main():
    os.chdir(ROOT)
    commit = git("rev-parse", "HEAD")
    clean_at(commit)
    lock = ROOT / ".evaluation-lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(
            "An evaluation is already running. Remove .evaluation-lock only if the previous process has stopped."
        )
    try:
        branch = git("symbolic-ref", "--short", "HEAD")
        remote_head = git("ls-remote", "--symref", "origin", "HEAD")
        default = next(
            (
                line.split()[1].removeprefix("refs/heads/")
                for line in remote_head.splitlines()
                if line.startswith("ref: ")
            ),
            None,
        )
        if branch != default:
            raise RuntimeError("Run this command on the origin default branch.")
        git("var", "GIT_AUTHOR_IDENT")
        git("var", "GIT_COMMITTER_IDENT")
        from evaluation.dashboard import END, START

        readme = (ROOT / "README.MD").read_text()
        if (
            readme.count(START) != 1
            or readme.count(END) != 1
            or readme.index(START) > readme.index(END)
        ):
            raise RuntimeError(
                "README dashboard markers must appear exactly once and in order."
            )
        # Install declared runtime requirements into the activated virtual environment.
        dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *dependencies], check=True
        )
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        if not os.environ.get("AI_CREDITS"):
            raise RuntimeError("Missing AI_CREDITS in the environment or local .env.")
        os.environ["LLM_PROVIDER"] = "aicredits"
        seed = secrets.randbits(32)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"-{seed}"
        run_dir = ROOT / "evaluation/runs" / run_id
        run_dir.mkdir(parents=True)
        clean_at(commit)
        subprocess.run([sys.executable, "-m", "src.rag.index"], check=True)
        print(
            "Starting 100 questions through AI Credits (paid calls, no retries).",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.run",
                "--output-dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--commit",
                commit,
            ],
            check=True,
        )
        # Prepare updates away from the working tree; preserve every manual README edit.
        staging = run_dir / "publish"
        staging.mkdir()
        shutil.copy(ROOT / "README.MD", staging / "README.MD")
        (staging / "evaluation").mkdir()
        if (ROOT / "evaluation/history.csv").exists():
            shutil.copy(
                ROOT / "evaluation/history.csv", staging / "evaluation/history.csv"
            )
        shutil.copy(run_dir / "results.csv", staging / "results.csv")
        update_dashboard(staging, json.loads((run_dir / "run.json").read_text()))
        clean_at(commit)
        paths = [
            "results.csv",
            "evaluation/history.csv",
            "assets/evaluation.svg",
            "README.MD",
        ]
        for name in paths:
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(staging / name, target)
        # Detect concurrent changes before staging the generated files.
        changed = set(git("diff", "--name-only", "HEAD").splitlines())
        untracked = set(git("ls-files", "--others", "--exclude-standard").splitlines())
        if (changed | untracked) - set(paths):
            raise RuntimeError("Other files changed; nothing committed or pushed.")
        for name in paths:
            if (ROOT / name).read_bytes() != (staging / name).read_bytes():
                raise RuntimeError("Generated file edited; nothing committed or pushed.")
        subprocess.run(["git", "add", "--", *paths], check=True)
        if git("rev-parse", "HEAD") != commit:
            raise RuntimeError(
                "HEAD changed before commit; generated files remain local. Nothing pushed."
            )
        subprocess.run(
            [
                "git",
                "commit",
                "--only",
                "-m",
                f"docs: evaluation {run_id}",
                "--",
                *paths,
            ],
            check=True,
        )
        print(
            "Evaluation committed. Pushing; on failure, retry git push without reevaluating.",
            flush=True,
        )
        subprocess.run(
            ["git", "push", "origin", f"HEAD:refs/heads/{default}"], check=True
        )
    finally:
        lock.rmdir()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc))
