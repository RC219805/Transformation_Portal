/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  outputFileTracingIncludes: {
    "/portal/assets/[...path]": ["../../config/portal_asset_manifest.json"]
  }
};

export default nextConfig;
