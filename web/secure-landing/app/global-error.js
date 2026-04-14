"use client";

import { FrontdoorErrorShell } from "../components/frontdoor-error-shell.js";

export const dynamic = "force-dynamic";

export default function GlobalError({ error, reset }) {
  return (
    <html lang="en">
      <body>
        <FrontdoorErrorShell
          title="The managed front door hit an unexpected failure."
          message="Retry the request. If the problem persists, inspect the runtime logs for the active front door instance."
          primaryAction={
            <button
              type="button"
              onClick={() => reset()}
              style={{
                minHeight: "44px",
                border: "1px solid #7dd3fc",
                borderRadius: "8px",
                padding: "0.8rem 1rem",
                background: "transparent",
                color: "#f8fafc",
                cursor: "pointer"
              }}
            >
              Retry
            </button>
          }
          reference={error?.digest ? String(error.digest) : ""}
        />
      </body>
    </html>
  );
}
