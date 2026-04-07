import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(MODULE_DIR, "../../..");
const MANIFEST_PATH = path.join(REPO_ROOT, "config", "portal_asset_manifest.json");

function loadPortalAssetManifest() {
  const raw = JSON.parse(readFileSync(MANIFEST_PATH, "utf-8"));
  const assets = raw?.assets;
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
