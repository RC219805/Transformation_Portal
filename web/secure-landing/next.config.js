import path from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = fileURLToPath(new URL(".", import.meta.url));
const repoRoot = path.resolve(appRoot, "../..");

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  outputFileTracingRoot: repoRoot,
  turbopack: {
    root: repoRoot
  },
  outputFileTracingIncludes: {
    "/portal/assets/[...path]": ["../../config/portal_asset_manifest.json"]
  }
};

export default nextConfig;
