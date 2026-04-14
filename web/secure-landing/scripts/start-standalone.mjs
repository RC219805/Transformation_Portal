import { existsSync } from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const FRONTDOOR_ROOT = path.resolve(path.dirname(SCRIPT_PATH), "..");

function resolveDistDir(frontdoorRoot = FRONTDOOR_ROOT, env = process.env) {
  const requestedDistDir = String(env?.TP_NEXT_DIST_DIR || "").trim();
  return path.resolve(frontdoorRoot, requestedDistDir || ".next");
}

function standaloneCandidates(frontdoorRoot = FRONTDOOR_ROOT, env = process.env) {
  const distDir = resolveDistDir(frontdoorRoot, env);
  return [
    path.join(distDir, "standalone", "server.js"),
    path.join(distDir, "standalone", "web", "secure-landing", "server.js")
  ];
}

export function resolveStandaloneServerPath(frontdoorRoot = FRONTDOOR_ROOT, { env = process.env } = {}) {
  for (const candidate of standaloneCandidates(frontdoorRoot, env)) {
    if (existsSync(candidate)) {
      return candidate;
    }
  }

  const standaloneRoot = path.join(resolveDistDir(frontdoorRoot, env), "standalone");
  throw new Error(
    `Managed frontdoor standalone build was not found under ${standaloneRoot}. Run npm run build under Node 22, then retry.`
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
