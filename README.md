# pandas-retrive-testgen

A small toolchain + UI to:
1) Fetch a target function spec from PyPI source (`fetch.py`)
2) Generate pytest tests using an LLM (`gen_tests.py`)
3) Evaluate test suite quality via mutation testing (`mutation_engine.py`)

UI stack:
- Backend: FastAPI (runs scripts and serves artifacts)
- Frontend: React + Vite (3-step wizard)

---

## Requirements

### Required
- Python **3.12+**
- Node.js **18+** (or 20+)
- npm

### LLM Provider (choose one)
- **OpenRouter (cloud)**: needs `OPENROUTER_API_KEY` (credits may be required)
- **Ollama (local/free)**: needs Ollama running on `127.0.0.1:11434`

---

## Project Structure

- `src/fetch.py`
- `src/gen_tests.py`
- `src/mutation_engine.py`
- `src/backend/main.py`
- `src/backend/requirements.txt`
- `src/frontend/`

Runtime artifacts:
- `src/backend/artifacts/<jobId>/...`

---

## Backend Setup (FastAPI)

```bash
cd src/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

OpenRouter (optional)

Set API key in the same terminal where you run the backend:

export OPENROUTER_API_KEY="YOUR_KEY_HERE"


If you get: OpenRouter HTTP 402 Insufficient credits, your OpenRouter account has no credits.

macOS SSL fix (only if you see CERTIFICATE_VERIFY_FAILED)

If you installed Python from python.org:
Applications → Python 3.12 → Install Certificates.command

Or set certifi path:

export SSL_CERT_FILE="$(python3 -c 'import certifi; print(certifi.where())')"


⸻

Run Backend

In src/backend (with venv activated):

python -m uvicorn main:app --reload --port 8000

Health check:
	•	http://localhost:8000/api/health

⸻

Frontend Setup (React + Vite)

Open a second terminal:

cd src/frontend
npm install


⸻

Run Frontend

npm run dev

Frontend URL:
	•	http://localhost:5173

The frontend proxies /api/* to http://localhost:8000 (see vite.config.js).

⸻

Using the UI

Step 1 — Fetch Spec
	•	Enter: pandas.melt (example)
	•	Click Fetch
	•	Output: spec.json inside artifacts folder

Step 2 — Generate Tests

Choose provider/model:

OpenRouter
	•	Default provider: openrouter
	•	Models: e.g. openrouter:anthropic/claude-3.5-sonnet
	•	Ensure OPENROUTER_API_KEY is set for the backend process

Ollama
	•	Default provider: ollama
	•	Models: ollama:llama3
	•	Ensure Ollama is running:

curl http://127.0.0.1:11434/api/tags



Click Generate Tests → test files appear under generated_tests/.

Step 3 — Mutation Testing
	•	Set pytest timeout (e.g. 60s)
	•	Click Run Mutation
	•	Output: JSON results under mutation_results/

All artifacts can be downloaded from the UI.

⸻

CLI Usage (Optional)

Fetch

python src/fetch.py pandas.melt

Generate (OpenRouter)

export OPENROUTER_API_KEY="YOUR_KEY_HERE"
python src/gen_tests.py \
  --src path/to/spec.json \
  --out-pattern path/to/generated_tests/{provider}_{model}.py \
  --models openrouter:anthropic/claude-3.5-sonnet \
  --default-provider openrouter \
  --temperature 0.0

Mutation

python -m pytest --version
python src/mutation_engine.py \
  --spec path/to/spec.json \
  --base-project-path path/to/base_project \
  --generated-dir generated_tests \
  --results-dir mutation_results \
  --pytest-timeout 60


⸻

