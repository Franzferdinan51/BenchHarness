"""Official SWE-bench docker grading via the `swebench` package.

Used by the SWE-family adapters when all three hold:
- `swebench` pip package importable (the `swe` extra),
- a reachable docker daemon,
- `BENCH_SWE_DOCKER` not set to `0`.

Otherwise adapters fall back to patch-overlap heuristics. Graded through
swebench's own run_evaluation so FAIL_TO_PASS / PASS_TO_PASS semantics are
exactly the official ones; works on Colima (amd64 images under emulation,
slower) as long as report dirs stay under $HOME.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


def docker_daemon_reachable() -> bool:
    try:
        import docker  # type: ignore

        docker.from_env().ping()
        return True
    except Exception:
        return False


def swebench_available() -> bool:
    try:
        import swebench.harness.run_evaluation
        import swebench.harness.utils  # noqa: F401
    except Exception:
        return False
    return True


def docker_grading_available() -> bool:
    if os.environ.get("BENCH_SWE_DOCKER", "").strip() == "0":
        return False
    return swebench_available() and docker_daemon_reachable()


# Task dataset -> eval-ready dataset (carries image/eval_script/log_parser).
# princeton-nlp/SWE-bench_Verified is the task-only mirror; the SWE-bench
# org upload has identical instance_ids plus eval columns (verified live).
EVAL_DATASETS = {
    "princeton-nlp/SWE-bench_Verified": "SWE-bench/SWE-bench_Verified",
    "SWE-bench/SWE-bench_Multilingual": "SWE-bench/SWE-bench_Multilingual",
}


def eval_dataset_for(dataset: str) -> str:
    return EVAL_DATASETS.get(dataset, dataset)


def instance_image(dataset: str, instance_id: str) -> str:
    """Docker image for an instance, from the eval-ready dataset row."""
    from datasets import load_dataset

    rows = load_dataset(eval_dataset_for(dataset), split="test")
    for row in rows:
        if row.get("instance_id") == instance_id:
            image = str(row.get("image", ""))
            if image:
                return image
            raise KeyError(f"no image column for {instance_id} in {dataset}")
    raise KeyError(f"{instance_id} not in {dataset}")


def ensure_image(image: str, timeout_secs: float = 1800.0) -> None:
    """Pull the instance image if missing; force amd64 on Apple Silicon.

    swebench pulls via docker-py without a platform, which 404s on arm64
    daemons for amd64-only images. Pre-pulling with the CLI flag fixes it.
    """
    have = subprocess.run(
        ["docker", "images", "-q", image], capture_output=True, text=True, check=False
    )
    if have.returncode == 0 and have.stdout.strip():
        return
    cmd = ["docker", "pull"]
    if platform.machine().lower() in ("arm64", "aarch64"):
        cmd += ["--platform", "linux/amd64"]
    cmd.append(image)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_secs, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker pull {image} failed: {proc.stderr[-400:]}")


@dataclass
class DockerGrade:
    resolved: bool
    details: str
    report_path: str = ""


def grade_with_docker(
    dataset: str,
    instance_id: str,
    model_patch: str,
    workdir: Path,
    timeout_secs: int = 1200,
) -> DockerGrade:
    """Run official FAIL_TO_PASS/PASS_TO_PASS grading for one instance."""
    from swebench.harness.run_evaluation import main as run_eval

    workdir.mkdir(parents=True, exist_ok=True)
    run_id = f"bh-{uuid.uuid4().hex[:8]}"
    predictions = {
        instance_id: {
            "instance_id": instance_id,
            "model_patch": model_patch,
            "model_name_or_path": "bench-harness",
        }
    }
    preds_path = workdir / f"{run_id}-preds.json"
    preds_path.write_text(json.dumps(predictions), encoding="utf-8")
    try:
        ensure_image(instance_image(dataset, instance_id))
    except Exception as exc:
        return DockerGrade(resolved=False, details=f"image prep failed: {exc}")
    try:
        report_path = run_eval(
            dataset_name=eval_dataset_for(dataset),
            split="test",
            instance_ids=[instance_id],
            predictions_path=str(preds_path),
            max_workers=1,
            open_file_limit=4096,
            run_id=run_id,
            timeout=timeout_secs,
            rewrite_reports=False,  # True = re-grade existing logs, runs nothing
            modal=False,
            report_dir=str(workdir),
        )
    except Exception as exc:
        return DockerGrade(
            resolved=False,
            details=f"swebench eval crashed: {type(exc).__name__}: {exc}",
        )
    try:
        report = json.loads(Path(str(report_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return DockerGrade(resolved=False, details=f"unreadable report: {exc}")
    resolved_ids = report.get("resolved_ids", report.get("resolved", [])) or []
    if instance_id in resolved_ids:
        return DockerGrade(
            resolved=True,
            details="FAIL_TO_PASS passed (official docker eval)",
            report_path=str(report_path),
        )
    for key, label in (
        ("error_ids", "eval error"),
        ("infra_failure_ids", "infra failure"),
        ("empty_patch_ids", "empty patch"),
        ("unresolved_ids", "unresolved"),
        ("unresolved", "unresolved"),
    ):
        if instance_id in (report.get(key, []) or []):
            return DockerGrade(
                resolved=False,
                details=f"{label} (official docker eval)",
                report_path=str(report_path),
            )
    return DockerGrade(
        resolved=False,
        details="not in report (see report file)",
        report_path=str(report_path),
    )
