import React, { useEffect, useMemo, useState } from "react";
import { postJson, getJson } from "./api.js";

function StepHeader({ step }) {
  const items = ["Fetch Spec", "Generate Tests", "Mutation"];
  return (
    <div className="stepper">
      {items.map((t, idx) => (
        <div key={t} className={"step " + (idx === step ? "active" : idx < step ? "done" : "")}>
          <div className="badge">{idx + 1}</div>
          <div className="label">{t}</div>
        </div>
      ))}
    </div>
  );
}

function JobBox({ title, jobId }) {
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    let alive = true;

    async function tick() {
      try {
        const j = await getJson(`/api/job/${jobId}`);
        if (!alive) return;
        setJob(j);
        setErr(null);
      } catch (e) {
        if (!alive) return;
        setErr(String(e.message || e));
      }
    }

    tick();
    const id = setInterval(tick, 1200);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [jobId]);

  if (!jobId) return null;

  return (
    <div className="card">
      <div className="cardTitle">
        <span>{title}</span>
        <span className={"pill " + (job?.status || "queued")}>{job?.status || "..."}</span>
      </div>

      {err ? <div className="error">Error: {err}</div> : null}

      <div className="subTitle">Job ID: {jobId}</div>

      <div className="grid2">
        <div>
          <div className="sectionTitle">Artifacts</div>
          <div className="artifactList">
            {(job?.artifacts || []).length === 0 ? <div className="muted">—</div> : null}
            {(job?.artifacts || []).map((a) => (
              <div key={a} className="artifactRow">
                <code>{a}</code>
                <a className="btn small" href={`/api/artifact/${jobId}/${a}`} target="_blank" rel="noreferrer">
                  Download
                </a>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="sectionTitle">Log</div>
          <pre className="log">{job?.log || ""}</pre>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [step, setStep] = useState(0);

  const [symbol, setSymbol] = useState("pandas.concat");

  const [fetchJobId, setFetchJobId] = useState("");
  const [genJobId, setGenJobId] = useState("");
  const [mutJobId, setMutJobId] = useState("");

  const [models, setModels] = useState("ollama:llama3");
  const [defaultProvider, setDefaultProvider] = useState("openrouter");
  const [temperature, setTemperature] = useState(0.0);
  const [stream, setStream] = useState(true);

  const [pytestTimeout, setPytestTimeout] = useState(60);

  const canGoGen = useMemo(() => Boolean(fetchJobId), [fetchJobId]);
  const canGoMut = useMemo(() => Boolean(fetchJobId && genJobId), [fetchJobId, genJobId]);

  async function doFetch() {
    const r = await postJson("/api/fetch", { symbol });
    setFetchJobId(r.jobId);
    setStep(0);
  }

  async function doGenerate() {
    if (!fetchJobId) return;
    const r = await postJson("/api/generate", {
      spec_job_id: fetchJobId,
      models,
      default_provider: defaultProvider,
      temperature,
      stream
    });
    setGenJobId(r.jobId);
    setStep(1);
  }

  async function doMutate() {
    if (!fetchJobId || !genJobId) return;
    const r = await postJson("/api/mutate", {
      spec_job_id: fetchJobId,
      generated_job_id: genJobId,
      pytest_timeout: pytestTimeout
    });
    setMutJobId(r.jobId);
    setStep(2);
  }

  return (
    <div className="wrap">
      <div className="top">
        <h1>Mutation Tool UI</h1>
        <p className="muted">
          Simple UI to run <code>fetch.py</code> → <code>gen_tests.py</code> → <code>mutation_engine.py</code>
        </p>
      </div>

      <StepHeader step={step} />

      <div className="card">
        <div className="cardTitle">Step 1 — Fetch Spec</div>
        <div className="row">
          <label>Qualified name</label>
          <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="pandas.concat" />
          <button className="btn" onClick={doFetch}>Fetch</button>
        </div>
      </div>

      <JobBox title="Fetch Job" jobId={fetchJobId} />

      <div className="card">
        <div className="cardTitle">Step 2 — Generate Tests</div>
        <div className={"muted " + (canGoGen ? "" : "disabled")}>
          Needs a completed Fetch job.
        </div>

        <div className="row">
          <label>Models (--models)</label>
          <input value={models} onChange={(e) => setModels(e.target.value)} placeholder="ollama:llama3" />
        </div>

        <div className="row">
          <label>Default provider</label>
          <select value={defaultProvider} onChange={(e) => setDefaultProvider(e.target.value)}>
            <option value="openrouter">openrouter</option>
            <option value="ollama">ollama</option>
          </select>
        </div>

        <div className="row">
          <label>Temperature</label>
          <input
            type="number"
            value={temperature}
            step="0.1"
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
          />
        </div>

        <div className="row">
          <label>Stream</label>
          <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
        </div>

        <div className="row">
          <span />
          <button className="btn" disabled={!canGoGen} onClick={doGenerate}>Generate Tests</button>
        </div>
      </div>

      <JobBox title="Generate Job" jobId={genJobId} />

      <div className="card">
        <div className="cardTitle">Step 3 — Mutation</div>
        <div className={"muted " + (canGoMut ? "" : "disabled")}>
          Needs completed Fetch + Generate jobs.
        </div>

        <div className="row">
          <label>pytest timeout (s)</label>
          <input
            type="number"
            value={pytestTimeout}
            onChange={(e) => setPytestTimeout(parseInt(e.target.value || "60", 10))}
          />
        </div>

        <div className="row">
          <span />
          <button className="btn" disabled={!canGoMut} onClick={doMutate}>Run Mutation</button>
        </div>
      </div>

      <JobBox title="Mutation Job" jobId={mutJobId} />

      <div className="footer muted">
        Tip: For more reliability, add a Baseline run step before mutation.
      </div>
    </div>
  );
}