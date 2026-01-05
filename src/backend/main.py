import os
import json
import uuid
import shutil
import tempfile
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- Ensure SSL certificates are available (macOS/python.org common issue)
try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    # If certifi isn't installed, SSL_CERT_FILE won't be auto-set.
    # In that case, rely on system/Install Certificates.command or shell env.
    pass

APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR.parent  # assumes scripts live one level above backend/
ARTIFACTS_DIR = APP_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FETCH_SCRIPT = SCRIPTS_DIR / "fetch.py"
GEN_SCRIPT = SCRIPTS_DIR / "gen_tests.py"
MUT_SCRIPT = SCRIPTS_DIR / "mutation_engine.py"

# Always use the same Python interpreter that runs the backend
PYTHON_EXE = sys.executable

# -------------------------
# Simple in-memory job store
# -------------------------
JOBS: Dict[str, Dict[str, Any]] = {}


def _new_job(kind: str) -> str:
    job_id = uuid.uuid4().hex
    job_dir = ARTIFACTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id] = {
        "id": job_id,
        "kind": kind,
        "status": "queued",  # queued | running | done | error
        "log": "",
        "artifacts": [],
        "job_dir": str(job_dir),
        "result": None,
        "error": None,
    }
    return job_id


def _append_log(job_id: str, text: str) -> None:
    JOBS[job_id]["log"] += text


def _run_cmd(
    job_id: str,
    cmd: List[str],
    cwd: Path,
    timeout_s: int = 600,
) -> subprocess.CompletedProcess:
    JOBS[job_id]["status"] = "running"
    _append_log(job_id, f"$ {' '.join(cmd)}\n")

    env = os.environ.copy()  # IMPORTANT: pass SSL_CERT_FILE etc.

    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
        _append_log(job_id, cp.stdout or "")
        _append_log(job_id, cp.stderr or "")
        if cp.returncode != 0:
            raise RuntimeError(f"Command failed with return code {cp.returncode}")
        return cp
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout_s}s")


def _list_artifacts(job_dir: Path) -> List[str]:
    files: List[str] = []
    for p in job_dir.rglob("*"):
        if p.is_file():
            files.append(str(p.relative_to(job_dir)))
    return sorted(files)


# -------------------------
# Request models
# -------------------------
class FetchReq(BaseModel):
    symbol: str = Field(..., examples=["pandas.concat"])


class GenerateReq(BaseModel):
    spec_job_id: str
    models: str = Field(..., examples=["ollama:llama3"])
    default_provider: str = Field("openrouter", pattern="^(openrouter|ollama)$")
    temperature: float = 0.0
    stream: bool = True


class MutateReq(BaseModel):
    spec_job_id: str
    generated_job_id: str
    pytest_timeout: int = 60


app = FastAPI(title="Mutation Tool UI Backend", version="0.1.0")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "python": PYTHON_EXE,
        "ssl_cert_file": os.environ.get("SSL_CERT_FILE"),
    }


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = Path(job["job_dir"])
    job["artifacts"] = _list_artifacts(job_dir)
    return job


@app.get("/api/artifact/{job_id}/{path:path}")
def download_artifact(job_id: str, path: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_dir = Path(job["job_dir"])
    file_path = job_dir / path
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(str(file_path))


@app.post("/api/fetch")
def run_fetch(req: FetchReq):
    job_id = _new_job("fetch")
    job_dir = Path(JOBS[job_id]["job_dir"])

    try:
        with tempfile.TemporaryDirectory(prefix="fetch_") as td:
            td_path = Path(td)
            cmd = [PYTHON_EXE, str(FETCH_SCRIPT), req.symbol]
            _run_cmd(job_id, cmd, cwd=td_path, timeout_s=900)

            out_name = req.symbol.replace(".", "_") + ".json"
            out_file = td_path / out_name
            if not out_file.exists():
                raise RuntimeError("fetch.py did not produce expected json output")

            shutil.copy2(out_file, job_dir / "spec.json")

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {"spec": "spec.json"}
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

    return {"jobId": job_id}


@app.post("/api/generate")
def run_generate(req: GenerateReq):
    spec_job = JOBS.get(req.spec_job_id)
    if not spec_job or spec_job.get("status") != "done":
        raise HTTPException(status_code=400, detail="spec_job_id is not ready")

    job_id = _new_job("generate")
    job_dir = Path(JOBS[job_id]["job_dir"])
    spec_path = Path(spec_job["job_dir"]) / "spec.json"

    try:
        shutil.copy2(spec_path, job_dir / "spec.json")

        out_dir = job_dir / "generated_tests"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_pattern = str(out_dir / "{provider}_{model}.py")

        cmd = [
            PYTHON_EXE, str(GEN_SCRIPT),
            "--src", str(job_dir / "spec.json"),
            "--out-pattern", out_pattern,
            "--models", req.models,
            "--default-provider", req.default_provider,
            "--temperature", str(req.temperature),
        ]
        if not req.stream:
            cmd.append("--no-stream")

        _run_cmd(job_id, cmd, cwd=job_dir, timeout_s=1800)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {"generated_dir": "generated_tests"}
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

    return {"jobId": job_id}


@app.post("/api/mutate")
def run_mutate(req: MutateReq):
    spec_job = JOBS.get(req.spec_job_id)
    gen_job = JOBS.get(req.generated_job_id)

    if not spec_job or spec_job.get("status") != "done":
        raise HTTPException(status_code=400, detail="spec_job_id is not ready")
    if not gen_job or gen_job.get("status") != "done":
        raise HTTPException(status_code=400, detail="generated_job_id is not ready")

    job_id = _new_job("mutate")
    job_dir = Path(JOBS[job_id]["job_dir"])

    try:
        base_proj = job_dir / "base_project"
        base_proj.mkdir(parents=True, exist_ok=True)

        src_gen_dir = Path(gen_job["job_dir"]) / "generated_tests"
        dst_gen_dir = base_proj / "generated_tests"
        shutil.copytree(src_gen_dir, dst_gen_dir, dirs_exist_ok=True)

        spec_path = Path(spec_job["job_dir"]) / "spec.json"
        shutil.copy2(spec_path, base_proj / "spec.json")

        results_dir = base_proj / "mutation_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            PYTHON_EXE, str(MUT_SCRIPT),
            "--spec", str(base_proj / "spec.json"),
            "--base-project-path", str(base_proj),
            "--generated-dir", "generated_tests",
            "--results-dir", "mutation_results",
            "--pytest-timeout", str(req.pytest_timeout),
        ]
        _run_cmd(job_id, cmd, cwd=base_proj, timeout_s=2400)

        shutil.copytree(results_dir, job_dir / "mutation_results", dirs_exist_ok=True)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {"results_dir": "mutation_results"}
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)

    return {"jobId": job_id}