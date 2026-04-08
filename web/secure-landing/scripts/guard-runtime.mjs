import { ensureSupportedRuntime } from "../lib/runtime-guard.js";

try {
  await ensureSupportedRuntime();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
