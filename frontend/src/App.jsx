import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";
import Settings from "./pages/Settings";
import { getCurrentUser, login, logout, register } from "./services/api";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("gitfix_token"));
  const [user, setUser] = useState(null);
  const [activeView, setActiveView] = useState("dashboard");
  const [checkingSession, setCheckingSession] = useState(Boolean(token));
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem("gitfix_dark") === "true");

  useEffect(() => {
    document.body.classList.toggle("dark-mode", darkMode);
    localStorage.setItem("gitfix_dark", darkMode);
  }, [darkMode]);

  useEffect(() => {
    if (!token) {
      setCheckingSession(false);
      return;
    }

    getCurrentUser(token)
      .then(setUser)
      .catch(() => {
        localStorage.removeItem("gitfix_token");
        setToken("");
        setUser(null);
      })
      .finally(() => setCheckingSession(false));
  }, [token]);

  async function handleLogin(username, password) {
    const session = await login(username, password);
    localStorage.setItem("gitfix_token", session.token);
    setToken(session.token);
    setUser({ username: session.username });
  }

  async function handleRegister(username, password) {
    const session = await register(username, password);
    localStorage.setItem("gitfix_token", session.token);
    setToken(session.token);
    setUser({ username: session.username });
  }

  async function handleLogout() {
    if (token) {
      await logout(token).catch(() => {});
    }
    localStorage.removeItem("gitfix_token");
    setToken("");
    setUser(null);
    setActiveView("dashboard");
  }

  if (checkingSession) {
    return <div className="loading-screen">Checking session...</div>;
  }

  if (!token) {
    return <Login onLogin={handleLogin} onRegister={handleRegister} />;
  }

  return (
    <div className="app-layout">
      <Sidebar
        activeView={activeView}
        onChangeView={setActiveView}
        user={user}
        onLogout={handleLogout}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode(!darkMode)}
      />
      <main className="content-shell">
        {activeView === "dashboard" ? <Dashboard token={token} /> : null}
        {activeView === "settings" ? <Settings user={user} /> : null}
      </main>
    </div>
  );
}
