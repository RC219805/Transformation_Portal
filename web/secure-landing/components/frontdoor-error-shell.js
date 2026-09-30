const SHELL_STYLE = {
  position: "fixed",
  inset: 0,
  overflow: "auto",
  display: "grid",
  placeItems: "center",
  padding: "1.25rem",
  background: "#111318",
  color: "#f5f6f8",
  fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  boxSizing: "border-box",
};

const PANEL_STYLE = {
  boxSizing: "border-box",
  width: "min(100%, 34rem)",
  border: "1px solid #343942",
  borderRadius: "12px",
  background: "#191c22",
  padding: "clamp(1.5rem, 5vw, 3rem)",
};

const META_STYLE = {
  margin: 0,
  fontSize: "0.6875rem",
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "#94b6ff",
};

const TITLE_STYLE = {
  margin: "1rem 0 0",
  fontSize: "clamp(1.8rem, 4vw, 2.5rem)",
  fontWeight: 500,
  letterSpacing: "-0.04em",
  lineHeight: 1.15,
};

const COPY_STYLE = {
  margin: "1rem 0 0",
  color: "#b6bbc5",
  fontSize: "0.9375rem",
  lineHeight: 1.75,
};

const ACTION_ROW_STYLE = {
  display: "flex",
  flexWrap: "wrap",
  gap: "0.75rem",
  marginTop: "2rem",
};

export const FRONTDOOR_ERROR_ACTION_STYLE = {
  boxSizing: "border-box",
  minHeight: "46px",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  border: "1px solid #343942",
  borderRadius: "8px",
  padding: "0.8rem 1rem",
  color: "#f5f6f8",
  background: "transparent",
  font: "inherit",
  fontSize: "0.875rem",
  fontWeight: 500,
  cursor: "pointer",
  textDecoration: "none",
};

export function FrontdoorErrorShell({
  title,
  message,
  primaryAction = null,
  reference = "",
  status = "Recovery",
}) {
  return (
    <main style={SHELL_STYLE} data-ui="frontdoor-error-shell">
      <section style={PANEL_STYLE} aria-labelledby="frontdoor-error-title">
        <p style={META_STYLE}>Dynamic Neural Access</p>
        <p style={{ ...META_STYLE, marginTop: "2.5rem", color: "#b6bbc5" }}>{status}</p>
        <h1 id="frontdoor-error-title" style={TITLE_STYLE}>{title}</h1>
        <p style={COPY_STYLE}>{message}</p>
        <div style={ACTION_ROW_STYLE}>
          {primaryAction}
          <a href="/" style={FRONTDOOR_ERROR_ACTION_STYLE}>
            Return home
          </a>
        </div>
        {reference ? (
          <p style={{ ...COPY_STYLE, fontSize: "0.75rem", overflowWrap: "anywhere" }}>
            Reference: {reference}
          </p>
        ) : null}
      </section>
    </main>
  );
}
