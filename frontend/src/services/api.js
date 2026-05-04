const API_BASE = "http://localhost:8000/api";

export async function startFix(issueUrl) {
  const response = await fetch(`${API_BASE}/fix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ issue_url: issueUrl }),
  });
  if (!response.ok) {
    throw new Error(`Failed to start fix: ${response.status}`);
  }
  return response.json();
}

export async function listRuns() {
  const response = await fetch(`${API_BASE}/runs`);
  if (!response.ok) {
    throw new Error(`Failed to fetch runs: ${response.status}`);
  }
  return response.json();
}
