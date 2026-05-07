import { useState } from "react";

export default function Login({ onLogin, onRegister }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const isRegistering = mode === "register";

  async function handleSubmit(event) {
    event.preventDefault();
    try {
      setLoading(true);
      setError("");
      if (isRegistering) {
        await onRegister(username.trim(), password);
      } else {
        await onLogin(username.trim(), password);
      }
    } catch (err) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setError("");
    setPassword("");
  }

  return (
    <main className="login-page">
      <form className="login-card" onSubmit={handleSubmit}>
        <div className="brand login-brand">
          <div className="brand-mark">G</div>
          <div>
            <div className="brand-title">GitFix AI</div>
            <div className="brand-subtitle">Secure Dashboard</div>
          </div>
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={mode === "login" ? "active" : ""}
            onClick={() => switchMode("login")}
          >
            Sign In
          </button>
          <button
            type="button"
            className={mode === "register" ? "active" : ""}
            onClick={() => switchMode("register")}
          >
            Register
          </button>
        </div>

        <label>
          Email or username
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete={isRegistering ? "email" : "username"}
          />
        </label>

        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
          />
        </label>

        {error ? <p className="error-text">{error}</p> : null}

        <button disabled={loading || !username.trim() || !password}>
          {loading
            ? isRegistering ? "Creating account..." : "Signing in..."
            : isRegistering ? "Create Account" : "Sign In"}
        </button>
      </form>
    </main>
  );
}
