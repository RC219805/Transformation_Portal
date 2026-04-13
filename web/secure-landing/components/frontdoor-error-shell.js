const SHELL_STYLE = {
  position: "fixed",
  inset: 0,
  overflow: "auto",
  display: "grid",
  placeItems: "center",
  padding: "2rem",
  background: "linear-gradient(180deg, #06111d 0%, #081523 48%, #06111d 100%)",
  color: "#f8fafc",
  fontFamily: "system-ui, sans-serif",
  boxSizing: "border-box",
};

const PANEL_STYLE = {
  width: "min(100%, 32rem)",
  border: "1px solid rgba(148, 163, 184, 0.24)",
  borderRadius: "8px",
  background: "rgba(8, 15, 29, 0.94)",
  boxShadow: "0 20px 44px rgba(2, 6, 23, 0.38)",
  padding: "1.5rem",
};

const META_STYLE = {
  margin: 0,
  fontSize: "0.75rem",
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  color: "#93c5fd",
};

const TITLE_STYLE = {
  margin: "0.75rem 0 0",
  fontSize: "2rem",
  lineHeight: 1.1,
};

const COPY_STYLE = {
  margin: "0.9rem 0 0",
  color: "#cbd5e1",
  lineHeight: 1.6,
};

const ACTION_ROW_STYLE = {
  display: "flex",
  flexWrap: "wrap",
  gap: "0.75rem",
  marginTop: "1.5rem",
};

const SECONDARY_ACTION_STYLE = {
  minHeight: "44px",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  border: "1px solid rgba(148, 163, 184, 0.24)",
  borderRadius: "8px",
  padding: "0.8rem 1rem",
  color: "#cbd5e1",
  textDecoration: "none",
};

export function FrontdoorErrorShell({
  title,
  message,
  primaryAction = null,
  reference = "",
}) {
  return (
    <main style={SHELL_STYLE}>
      <section style={PANEL_STYLE}>
        <p style={META_STYLE}>Dynamic Neural Access</p>
        <h1 style={TITLE_STYLE}>{title}</h1>
        <p style={COPY_STYLE}>{message}</p>
        <div style={ACTION_ROW_STYLE}>
          {primaryAction}
          <a href="/" style={SECONDARY_ACTION_STYLE}>
            Return home
          </a>
        </div>
        {reference ? (
          <p style={{ ...COPY_STYLE, fontSize: "0.9rem", color: "#94a3b8" }}>
            Reference: {reference}
          </p>
        ) : null}
      </section>
    </main>
  );
}
