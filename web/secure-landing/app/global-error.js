"use client";

import { FrontdoorErrorShell, FRONTDOOR_ERROR_ACTION_STYLE } from "../components/frontdoor-error-shell.js";

export const dynamic = "force-dynamic";

export default function GlobalError({ error, reset }) {
  return (
    <html lang="en">
      <body>
        <FrontdoorErrorShell
          title="We couldn't load this page."
          message="Try again, or return to the overview to reopen your workspace. If the problem continues, contact your workspace administrator."
          primaryAction={
            <button
              type="button"
              onClick={() => reset()}
              style={{
                ...FRONTDOOR_ERROR_ACTION_STYLE,
                borderColor: "#2458d3",
                background: "#2458d3",
                color: "#ffffff"
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
