"use client";

export const dynamic = "force-dynamic";

export default function GlobalError({ error, reset }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          padding: "2rem",
          background: "#08111a",
          color: "#f5f7fb",
          fontFamily: "system-ui, sans-serif"
        }}
      >
        <main style={{ maxWidth: "40rem" }}>
          <p
            style={{
              margin: 0,
              fontSize: "0.8rem",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "#8bc5ff"
            }}
          >
            Dynamic Neural Access
          </p>
          <h1 style={{ margin: "0.75rem 0 0", fontSize: "2rem", lineHeight: 1.1 }}>
            The managed front door hit an unexpected failure.
          </h1>
          <p style={{ margin: "1rem 0 0", color: "#c5d1de", lineHeight: 1.6 }}>
            Retry the request. If the problem persists, inspect the runtime logs for the active
            front door instance.
          </p>
          <button
            type="button"
            onClick={() => reset()}
            style={{
              marginTop: "1.5rem",
              border: "1px solid #8bc5ff",
              borderRadius: "999px",
              padding: "0.8rem 1.25rem",
              background: "transparent",
              color: "inherit",
              cursor: "pointer"
            }}
          >
            Retry
          </button>
          {error?.digest ? (
            <p style={{ margin: "1rem 0 0", color: "#9aa8b7", fontSize: "0.9rem" }}>
              Reference: {error.digest}
            </p>
          ) : null}
        </main>
      </body>
    </html>
  );
}
