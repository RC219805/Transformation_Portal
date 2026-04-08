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

function formatStandaloneErrorDetail(error) {
  return error instanceof Error ? error.message : String(error);
}

export function formatStandaloneSpawnFailureMessage(serverPath, error) {
  return `Managed frontdoor standalone server could not start from ${serverPath}: ${formatStandaloneErrorDetail(error)}. Run npm run build under Node 22, verify the standalone output is readable and executable, then retry.`;
}

export function formatStandaloneSignalRelayFailureMessage(signal, error) {
  return `Managed frontdoor standalone server exited with signal ${signal}, but the launcher could not relay that signal cleanly: ${formatStandaloneErrorDetail(error)}. Stop the child process manually and retry if needed.`;
}

export function attachStandaloneLifecycleHandlers(
  child,
  {
    serverPath,
    processLike = process,
    consoleLike = console
  }
) {
  child.on("error", (error) => {
    consoleLike.error(formatStandaloneSpawnFailureMessage(serverPath, error));
    processLike.exit(1);
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      try {
        processLike.kill(processLike.pid, signal);
      } catch (error) {
        consoleLike.error(formatStandaloneSignalRelayFailureMessage(signal, error));
        processLike.exit(1);
      }
      return;
    }
    processLike.exit(code ?? 0);
  });

  return child;
}

export function spawnStandaloneServer(
  serverPath,
  argv = process.argv.slice(2),
  {
    spawnImpl = spawn,
    frontdoorRoot = FRONTDOOR_ROOT,
    processLike = process,
    consoleLike = console
  } = {}
) {
  const child = spawnImpl(processLike.execPath ?? process.execPath, [serverPath, ...argv], {
    cwd: frontdoorRoot,
    env: processLike.env ?? process.env,
    stdio: "inherit"
  });

  return attachStandaloneLifecycleHandlers(child, {
    serverPath,
    processLike,
    consoleLike
  });
}

export async function main(argv = process.argv.slice(2)) {
  const serverPath = resolveStandaloneServerPath();
  spawnStandaloneServer(serverPath, argv);
}

if (process.argv[1] && path.resolve(process.argv[1]) === SCRIPT_PATH) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
