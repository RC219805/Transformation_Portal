import { FrontdoorErrorShell } from "../components/frontdoor-error-shell.js";

export const dynamic = "force-dynamic";

export default function NotFound() {
  return (
    <FrontdoorErrorShell
      status="404 · Page not found"
      title="This page couldn't be found."
      message="The requested front door route was not found. Check the address, or return to the overview to continue."
    />
  );
}
