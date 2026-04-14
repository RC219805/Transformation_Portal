import { FrontdoorErrorShell } from "../components/frontdoor-error-shell.js";

export const dynamic = "force-dynamic";

export default function NotFound() {
  return (
    <FrontdoorErrorShell
      title="The requested front door route was not found."
      message="Check the managed entry path and retry from the secure landing surface."
    />
  );
}
