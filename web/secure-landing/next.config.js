import path from "node:path";
import { fileURLToPath } from "node:url";
import nextConstants from "next/constants.js";

const appRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = path.resolve(appRoot, "../..");
const requestedDistDir = String(process.env.TP_NEXT_DIST_DIR || "").trim();
const { PHASE_DEVELOPMENT_SERVER } = nextConstants;

/** @type {import('next').NextConfig} */
function nextConfig(phase) {
  const usesVercelBuildOutput = process.env.VERCEL === "1";
  // Vercel's adapter emits .next/output and omits the server trace that
  // Next 16.3 otherwise expects while copying standalone output.
  const standaloneConfig = phase === PHASE_DEVELOPMENT_SERVER
    ? {}
    : {
        ...(usesVercelBuildOutput ? {} : { output: "standalone" }),
        outputFileTracingRoot: repoRoot,
        outputFileTracingIncludes: {
          "/portal/assets/[...path]": ["../../config/portal_asset_manifest.json"]
        }
      };

  return {
    distDir: requestedDistDir || ".next",
    turbopack: {
      root: repoRoot
    },
    ...standaloneConfig
  };
}

export default nextConfig;
