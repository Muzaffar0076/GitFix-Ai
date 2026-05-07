export default function Settings({ user }) {
  return (
    <div>
      <h1 className="title">Settings</h1>
      <p className="subtitle">Basic account and runtime details.</p>

      <div className="card">
        <h3>Signed In User</h3>
        <p className="muted">Username</p>
        <p className="setting-value">{user?.username || "-"}</p>
      </div>

      <div className="card">
        <h3>Authentication</h3>
        <p className="muted">
          The dashboard uses a bearer token from the backend login endpoint.
          Tokens are kept in browser storage and API calls include the token in
          the Authorization header.
        </p>
      </div>
    </div>
  );
}
