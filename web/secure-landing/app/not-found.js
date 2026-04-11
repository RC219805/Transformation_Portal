export const dynamic = "force-dynamic";

export default function NotFound() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: "2rem",
        background: "#08111a",
        color: "#f5f7fb",
        fontFamily: "system-ui, sans-serif"
      }}
    >
      <div style={{ maxWidth: "36rem" }}>
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
          The requested front door route was not found.
        </h1>
        <p style={{ margin: "1rem 0 0", color: "#c5d1de", lineHeight: 1.6 }}>
          Check the managed entry path and retry from the secure landing surface.
        </p>
      </div>
    </main>
  );
}
