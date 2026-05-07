export default function Sidebar({ activeView, onChangeView, user, onLogout }) {
  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: "D" },
    { id: "settings", label: "Settings", icon: "S" },
  ];

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">G</div>
        <div>
          <div className="brand-title">GitFix AI</div>
          <div className="brand-subtitle">Agent Console</div>
        </div>
      </div>

      <nav className="nav-list">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeView === item.id ? "active" : ""}`}
            onClick={() => onChangeView(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div>
          <div className="user-name">{user?.username || "User"}</div>
          <div className="muted">Authenticated</div>
        </div>
        <button className="secondary-button" onClick={onLogout}>
          Logout
        </button>
      </div>
    </aside>
  );
}
