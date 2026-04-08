import { readFileSync } from "node:fs";
import path from "node:path";

const MANIFEST_CANDIDATES = Object.freeze([
  path.resolve(process.cwd(), "../../config/portal_asset_manifest.json"),
  path.resolve(process.cwd(), "config/portal_asset_manifest.json")
]);

function loadPortalAssetManifest() {
  let rawPortalAssetManifest = null;
  for (const candidatePath of MANIFEST_CANDIDATES) {
    try {
      rawPortalAssetManifest = JSON.parse(readFileSync(candidatePath, "utf-8"));
      break;
    } catch {
      // Keep searching known runtime locations.
    }
  }

  if (!rawPortalAssetManifest) {
    throw new Error("unable to load portal asset manifest from known runtime locations");
  }

  const assets = rawPortalAssetManifest?.assets;
  if (!assets || typeof assets !== "object" || Array.isArray(assets)) {
    throw new Error("portal asset manifest must define an assets object");
  }

  return Object.freeze({ ...assets });
}

export const PORTAL_ASSET_MANIFEST = loadPortalAssetManifest();
export const PORTAL_ASSET_PATHS = Object.freeze(Object.keys(PORTAL_ASSET_MANIFEST));

export function isAllowedPortalAssetPath(assetPath) {
  return Object.prototype.hasOwnProperty.call(PORTAL_ASSET_MANIFEST, assetPath);
}
