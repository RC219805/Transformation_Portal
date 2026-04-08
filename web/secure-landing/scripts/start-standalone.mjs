import { existsSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const FRONTDOOR_ROOT = path.resolve(path.dirname(SCRIPT_PATH), "..");

function standaloneCandidates(frontdoorRoot = FRONTDOOR_ROOT) {
  return [
    path.join(frontdoorRoot, ".next", "standalone", "server.js"),
    path.join(frontdoorRoot, ".next", "standalone", "web", "secure-landing", "server.js")
  ];
}

export function resolveStandaloneServerPath(frontdoorRoot = FRONTDOOR_ROOT) {
  for (const candidate of standaloneCandidates(frontdoorRoot)) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  throw new Error(
    `Managed frontdoor standalone build was not found under ${path.join(frontdoorRoot, ".next", "standalone")}. Run npm run build under Node 22, then retry.`
  );
}

export async function main(argv = process.argv.slice(2)) {
  const serverPath = resolveStandaloneServerPath();
  const child = spawn(process.execPath, [serverPath, ...argv], {
    cwd: FRONTDOOR_ROOT,
    env: process.env,
    stdio: "inherit"
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT_PATH) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
