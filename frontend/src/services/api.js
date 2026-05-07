const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function requestJson(url, options, fallbackMessage) {
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      throw new Error(fallbackMessage(response));
    }
    return response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("Backend is not running. Start FastAPI on port 8000.");
    }
    throw error;
  }
}

export async function login(username, password) {
  return requestJson(
    `${API_BASE}/auth/login`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    },
    () => "Invalid email/username or password",
  );
}

export async function register(username, password) {
  return requestJson(
    `${API_BASE}/auth/register`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    },
    (response) => response.status === 409
      ? "Email/username already registered"
      : "Registration failed",
  );
}

export async function logout(token) {
  await fetch(`${API_BASE}/auth/logout`, {
    method: "POST",
    headers: authHeaders(token),
  });
}

export async function getCurrentUser(token) {
  const response = await fetch(`${API_BASE}/auth/me`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    throw new Error("Session expired");
  }
  return response.json();
}

export async function startFix(issueUrl, token) {
  const response = await fetch(`${API_BASE}/fix`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ issue_url: issueUrl }),
  });
  if (!response.ok) {
    throw new Error(`Failed to start fix: ${response.status}`);
  }
  return response.json();
}

export async function listRuns(token) {
  const response = await fetch(`${API_BASE}/runs`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    throw new Error(`Failed to fetch runs: ${response.status}`);
  }
  return response.json();
}
