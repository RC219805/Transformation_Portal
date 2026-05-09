import path from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = path.resolve(appRoot, "../..");
const requestedDistDir = String(process.env.TP_NEXT_DIST_DIR || "").trim();

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  distDir: requestedDistDir || ".next",
  outputFileTracingRoot: repoRoot,
  outputFileTracingIncludes: {
    "/portal/assets/[...path]": ["../../config/portal_asset_manifest.json"]
  }
};

export default nextConfig;
