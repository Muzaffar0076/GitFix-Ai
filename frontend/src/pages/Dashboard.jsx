import { useEffect, useMemo, useRef, useState } from "react";
import { listRuns, startFix } from "../services/api";

function formatTime(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

export default function Dashboard({ token }) {
  const [issueUrl, setIssueUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState([]);
  const [showAllRuns, setShowAllRuns] = useState(false);
  const [activeRunId, setActiveRunId] = useState("");
  const [liveLogs, setLiveLogs] = useState([]);
  const [error, setError] = useState("");
  const wsRef = useRef(null);

  const activeRun = useMemo(
    () => runs.find((run) => run.run_id === activeRunId) || null,
    [runs, activeRunId]
  );

  async function refreshRuns() {
    const data = await listRuns(token);
    setRuns(data);
  }

  function connectLogs(runId) {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setLiveLogs([]);
    const ws = new WebSocket(
      `ws://localhost:8000/api/ws/runs/${runId}?token=${encodeURIComponent(token)}`
    );
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "run_complete") {
          refreshRuns().catch(() => {});
          return;
        }
        setLiveLogs((prev) => [...prev, payload]);
      } catch {
        // ignore malformed payloads
      }
    };
    wsRef.current = ws;
  }

  async function handleStart() {
    if (!issueUrl.trim()) return;
    try {
      setLoading(true);
      setError("");
      const run = await startFix(issueUrl.trim(), token);
      setActiveRunId(run.run_id);
      connectLogs(run.run_id);
      await refreshRuns();
      setIssueUrl("");
    } catch (err) {
      setError(err.message || "Failed to run pipeline");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshRuns().catch(() => {});
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [token]);

  return (
    <div className="app-shell">
      <h1 className="title">GitFix AI Dashboard</h1>
      <p className="subtitle">Run issue-fix pipelines and monitor logs live.</p>

      <div className="card">
        <div className="row">
          <input
            value={issueUrl}
            onChange={(e) => setIssueUrl(e.target.value)}
            placeholder="https://github.com/owner/repo/issues/123"
          />
          <button disabled={loading} onClick={handleStart}>
            {loading ? "Running..." : "Start Fix"}
          </button>
        </div>
        {error ? <p className="muted">Error: {error}</p> : null}
      </div>

      <div className="grid">
        <div className="card">
          <h3>Runs</h3>
          {runs.length === 0 ? <p className="muted">No runs yet.</p> : null}
          <div className="runs-list">
            {(showAllRuns ? runs : runs.slice(0, 2)).map((run) => (
              <div className="run-item" key={run.run_id}>
                <div>
                  <strong>{run.status}</strong> · {run.run_id.slice(0, 8)}
                </div>
                <div className="muted">{run.issue_url}</div>
                <div className="muted">Created: {formatTime(run.created_at)}</div>
                <div className="row">
                  <button
                    onClick={() => {
                      setActiveRunId(run.run_id);
                      connectLogs(run.run_id);
                    }}
                  >
                    Watch Logs
                  </button>
                </div>
              </div>
            ))}
          </div>
          {runs.length > 2 && (
            <button 
              className="secondary-button" 
              style={{ width: "100%" }}
              onClick={() => setShowAllRuns(!showAllRuns)}
            >
              {showAllRuns ? "Show Less" : `See More (${runs.length - 2} more)`}
            </button>
          )}
        </div>

        <div className="card">
          <h3>Live Logs</h3>
          {activeRun ? (
            <p className="muted">
              Active Run: {activeRun.run_id} · Status: {activeRun.status}
            </p>
          ) : (
            <p className="muted">Select a run to stream logs.</p>
          )}
          <div className="logs">
            {liveLogs.length === 0 ? "No log events yet..." : null}
            {liveLogs.map((log, idx) => (
              <div key={`${log.timestamp}-${idx}`}>
                [{log.stage}] {log.message}
              </div>
            ))}
          </div>
          {activeRun?.pr_url ? (
            <p>
              PR:{" "}
              <a href={activeRun.pr_url} target="_blank" rel="noreferrer">
                {activeRun.pr_url}
              </a>
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}
