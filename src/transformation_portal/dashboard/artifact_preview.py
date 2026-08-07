"""Artifact preview API with type detection and streaming.

This module provides FastAPI endpoints for previewing artifacts
including images, 3D meshes, text, and JSON data.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformation_portal.storage.cas_store import ArtifactStore

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


# Global CAS reference
_global_cas: Optional["ArtifactStore"] = None  # type: ignore


def set_preview_cas(cas: "ArtifactStore") -> None:  # type: ignore
    """Set the global CAS for artifact preview.

    Args:
        cas: ArtifactStore instance
    """
    global _global_cas
    _global_cas = cas


# Extended MIME type mappings
MIME_EXTENSIONS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".exr": "image/x-exr",
    ".hdr": "image/vnd.radiance",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".obj": "model/obj",
    ".ply": "model/ply",
    ".stl": "model/stl",
    ".fbx": "model/fbx",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".safetensors": "application/x-safetensors",
    ".pt": "application/x-pytorch",
    ".pth": "application/x-pytorch",
    ".bin": "application/octet-stream",
}


def detect_content_type(path: Path, data: Optional[bytes] = None) -> str:
    """Detect content type from file extension and magic bytes.

    Args:
        path: File path
        data: Optional file header bytes

    Returns:
        MIME type string
    """
    # Try extension first
    ext = path.suffix.lower()
    if ext in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[ext]

    # Try mimetypes
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime

    # Try magic bytes if data provided
    if data:
        # PNG
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        # JPEG
        if data[:2] == b"\xff\xd8":
            return "image/jpeg"
        # GIF
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        # WebP
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        # GLB (glTF binary)
        if data[:4] == b"glTF":
            return "model/gltf-binary"
        # JSON
        if data[:1] in (b"{", b"["):
            try:
                data[:100].decode("utf-8")
                return "application/json"
            except:
                pass

    return "application/octet-stream"


def create_preview_router() -> "APIRouter":
    """Create the artifact preview router.

    Returns:
        FastAPI APIRouter with preview endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for artifact preview")

    router = APIRouter(prefix="/api/preview", tags=["preview"])

    def _resolve_path(hash: str) -> Optional[Path]:
        """Resolve CAS path for hash."""
        if _global_cas:
            try:
                obj = _global_cas.get_object(hash)
            except ValueError:
                return None
            if obj:
                return obj.path
        return None

    @router.get("/artifact/{hash}/meta")
    async def artifact_meta(hash: str):
        """Get artifact metadata including detected type.

        Args:
            hash: SHA-256 hash
        """
        path = _resolve_path(hash)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Read header for type detection
        with path.open("rb") as f:
            header = f.read(512)

        content_type = detect_content_type(path, header)
        size = path.stat().st_size

        # Determine preview capability
        previewable = content_type.startswith(("image/", "text/", "model/")) or content_type in (
            "application/json",
            "application/x-yaml",
        )

        return JSONResponse(
            {
                "hash": hash,
                "size_bytes": size,
                "content_type": content_type,
                "previewable": previewable,
                "is_image": content_type.startswith("image/"),
                "is_3d": content_type.startswith("model/"),
                "is_text": content_type.startswith("text/") or content_type == "application/json",
            }
        )

    @router.get("/artifact/{hash}/raw")
    async def artifact_raw(hash: str):
        """Stream raw artifact content.

        Args:
            hash: SHA-256 hash
        """
        path = _resolve_path(hash)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Read header for type detection
        with path.open("rb") as f:
            header = f.read(512)

        content_type = detect_content_type(path, header)

        return FileResponse(
            path=str(path),
            media_type=content_type,
            filename=f"{hash[:16]}{path.suffix or '.bin'}",
        )

    @router.get("/artifact/{hash}/thumbnail")
    async def artifact_thumbnail(hash: str, size: int = 256):
        """Generate thumbnail for image artifact.

        Args:
            hash: SHA-256 hash
            size: Maximum dimension
        """
        path = _resolve_path(hash)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Check if image
        with path.open("rb") as f:
            header = f.read(512)

        content_type = detect_content_type(path, header)
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Not an image")

        try:
            import io

            from PIL import Image

            img = Image.open(path)
            img.thumbnail((size, size))

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            return Response(
                content=buffer.read(),
                media_type="image/png",
            )
        except ImportError:
            # Pillow not available, return original
            return FileResponse(path=str(path), media_type=content_type)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @router.get("/artifact/{hash}/text")
    async def artifact_text(hash: str, max_chars: int = 10000):
        """Get text content of artifact.

        Args:
            hash: SHA-256 hash
            max_chars: Maximum characters to return
        """
        path = _resolve_path(hash)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        try:
            with path.open("r", encoding="utf-8") as f:
                content = f.read(max_chars)
                truncated = len(content) >= max_chars

            return JSONResponse(
                {
                    "hash": hash,
                    "content": content,
                    "truncated": truncated,
                }
            )
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Not a text file")

    @router.get("/", response_class=HTMLResponse)
    async def preview_viewer():
        """Serve the artifact preview viewer UI."""
        return get_preview_viewer_html()

    return router


def get_preview_viewer_html() -> str:
    """Get the artifact preview viewer HTML with 3D support."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artifact Preview</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/OBJLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/PLYLoader.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .header {
            background: #16213e;
            padding: 1rem 2rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.25rem; }
        .search {
            display: flex;
            gap: 0.5rem;
        }
        .search input {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            width: 400px;
            font-family: monospace;
        }
        .search button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .container { padding: 2rem; max-width: 1400px; margin: 0 auto; }
        .meta-panel {
            background: #16213e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: none;
        }
        .meta-panel.active { display: block; }
        .meta-row {
            display: flex;
            gap: 2rem;
            font-size: 0.875rem;
        }
        .meta-item {
            display: flex;
            gap: 0.5rem;
        }
        .meta-label { color: #94a3b8; }
        .preview-container {
            background: #16213e;
            border-radius: 0.5rem;
            overflow: hidden;
            min-height: 500px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .preview-container.image-mode img {
            max-width: 100%;
            max-height: 80vh;
            object-fit: contain;
        }
        .preview-container.text-mode {
            padding: 1rem;
            align-items: flex-start;
            justify-content: flex-start;
        }
        .preview-container.text-mode pre {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
            white-space: pre-wrap;
            word-break: break-all;
            max-height: 70vh;
            overflow: auto;
            width: 100%;
        }
        .preview-container.mesh-mode {
            position: relative;
        }
        .preview-container.mesh-mode canvas {
            width: 100% !important;
            height: 500px !important;
        }
        .mesh-controls {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .mesh-controls button {
            background: rgba(15, 52, 96, 0.9);
            border: none;
            color: #eee;
            padding: 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.75rem;
        }
        .mesh-controls button:hover { background: #e94560; }
        .placeholder {
            color: #94a3b8;
            text-align: center;
        }
        .placeholder h2 { margin-bottom: 0.5rem; }
        .error { color: #ff5252; }
        .loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #0f3460;
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Artifact Preview</h1>
        <div class="search">
            <input type="text" id="hash-input" placeholder="Enter artifact hash...">
            <button onclick="loadArtifact()">Preview</button>
        </div>
    </div>
    <div class="container">
        <div id="meta-panel" class="meta-panel">
            <div class="meta-row">
                <div class="meta-item">
                    <span class="meta-label">Hash:</span>
                    <span id="meta-hash">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Type:</span>
                    <span id="meta-type">-</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Size:</span>
                    <span id="meta-size">-</span>
                </div>
                <div class="meta-item">
                    <a id="download-link" href="#" style="color:#e94560;">Download</a>
                </div>
            </div>
        </div>
        <div id="preview-container" class="preview-container">
            <div class="placeholder">
                <h2>No Artifact Selected</h2>
                <p>Enter a hash above to preview an artifact</p>
            </div>
        </div>
    </div>

    <script>
        let currentHash = null;
        let threeScene = null;
        let threeRenderer = null;
        let threeCamera = null;
        let threeControls = null;
        let animationId = null;

        // Get hash from URL if present
        const urlParams = new URLSearchParams(window.location.search);
        const hashParam = urlParams.get('hash');
        if (hashParam) {
            document.getElementById('hash-input').value = hashParam;
            setTimeout(loadArtifact, 100);
        }

        async function loadArtifact() {
            const hash = document.getElementById('hash-input').value.trim();
            if (!hash) return;

            currentHash = hash;
            showLoading();

            try {
                const res = await fetch(`/api/preview/artifact/${hash}/meta`);
                if (!res.ok) throw new Error('Artifact not found');
                const meta = await res.json();

                showMeta(meta);

                if (meta.is_image) {
                    renderImage(hash);
                } else if (meta.is_3d) {
                    renderMesh(hash, meta.content_type);
                } else if (meta.is_text) {
                    renderText(hash);
                } else {
                    renderUnsupported(meta);
                }
            } catch (e) {
                showError(e.message);
            }
        }

        function showLoading() {
            const container = document.getElementById('preview-container');
            container.className = 'preview-container';
            container.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading artifact...</p>
                </div>
            `;
        }

        function showError(message) {
            const container = document.getElementById('preview-container');
            container.className = 'preview-container';
            container.innerHTML = `
                <div class="placeholder error">
                    <h2>Error</h2>
                    <p>${message}</p>
                </div>
            `;
        }

        function showMeta(meta) {
            document.getElementById('meta-panel').className = 'meta-panel active';
            document.getElementById('meta-hash').textContent = meta.hash;
            document.getElementById('meta-type').textContent = meta.content_type;
            document.getElementById('meta-size').textContent = formatSize(meta.size_bytes);
            document.getElementById('download-link').href = `/api/preview/artifact/${meta.hash}/raw`;
        }

        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
            return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
        }

        function renderImage(hash) {
            cleanup3D();
            const container = document.getElementById('preview-container');
            container.className = 'preview-container image-mode';

            const img = document.createElement('img');
            img.src = `/api/preview/artifact/${hash}/raw`;
            img.onerror = () => showError('Failed to load image');

            container.innerHTML = '';
            container.appendChild(img);
        }

        async function renderText(hash) {
            cleanup3D();
            const container = document.getElementById('preview-container');
            container.className = 'preview-container text-mode';

            try {
                const res = await fetch(`/api/preview/artifact/${hash}/text`);
                const data = await res.json();

                let content = data.content;
                // Try to format JSON
                if (content.trim().startsWith('{') || content.trim().startsWith('[')) {
                    try {
                        content = JSON.stringify(JSON.parse(content), null, 2);
                    } catch {}
                }

                container.innerHTML = `<pre>${escapeHtml(content)}${data.truncated ? '\\n\\n[truncated]' : ''}</pre>`;
            } catch (e) {
                showError('Failed to load text: ' + e.message);
            }
        }

        function renderMesh(hash, contentType) {
            cleanup3D();
            const container = document.getElementById('preview-container');
            container.className = 'preview-container mesh-mode';
            container.innerHTML = `
                <div class="mesh-controls">
                    <button onclick="resetCamera()">Reset View</button>
                    <button onclick="toggleWireframe()">Wireframe</button>
                    <button onclick="toggleGrid()">Grid</button>
                </div>
            `;

            // Setup Three.js scene
            threeScene = new THREE.Scene();
            threeScene.background = new THREE.Color(0x1a1a2e);

            threeCamera = new THREE.PerspectiveCamera(75, container.clientWidth / 500, 0.1, 1000);
            threeCamera.position.set(2, 2, 2);

            threeRenderer = new THREE.WebGLRenderer({ antialias: true });
            threeRenderer.setSize(container.clientWidth, 500);
            threeRenderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(threeRenderer.domElement);

            // Controls
            threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
            threeControls.enableDamping = true;
            threeControls.dampingFactor = 0.05;

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            threeScene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(5, 10, 7.5);
            threeScene.add(directionalLight);

            const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight2.position.set(-5, -5, -5);
            threeScene.add(directionalLight2);

            // Grid
            const grid = new THREE.GridHelper(10, 10, 0x0f3460, 0x0f3460);
            grid.name = 'grid';
            threeScene.add(grid);

            // Load model
            const url = `/api/preview/artifact/${hash}/raw`;

            if (contentType.includes('gltf')) {
                const loader = new THREE.GLTFLoader();
                loader.load(url,
                    (gltf) => {
                        centerAndScaleModel(gltf.scene);
                        threeScene.add(gltf.scene);
                    },
                    undefined,
                    (error) => showError('Failed to load model: ' + error.message)
                );
            } else if (contentType.includes('obj')) {
                const loader = new THREE.OBJLoader();
                loader.load(url,
                    (obj) => {
                        centerAndScaleModel(obj);
                        threeScene.add(obj);
                    },
                    undefined,
                    (error) => showError('Failed to load model')
                );
            } else if (contentType.includes('ply')) {
                const loader = new THREE.PLYLoader();
                loader.load(url,
                    (geometry) => {
                        geometry.computeVertexNormals();
                        const material = new THREE.MeshStandardMaterial({ color: 0xe94560, flatShading: true });
                        const mesh = new THREE.Mesh(geometry, material);
                        centerAndScaleModel(mesh);
                        threeScene.add(mesh);
                    },
                    undefined,
                    (error) => showError('Failed to load model')
                );
            }

            // Animation loop
            function animate() {
                animationId = requestAnimationFrame(animate);
                threeControls.update();
                threeRenderer.render(threeScene, threeCamera);
            }
            animate();

            // Handle resize
            window.addEventListener('resize', onWindowResize);
        }

        function centerAndScaleModel(object) {
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;

            object.position.sub(center);
            object.scale.multiplyScalar(scale);
        }

        function cleanup3D() {
            if (animationId) {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
            if (threeRenderer) {
                threeRenderer.dispose();
                threeRenderer = null;
            }
            threeScene = null;
            threeCamera = null;
            threeControls = null;
            window.removeEventListener('resize', onWindowResize);
        }

        function onWindowResize() {
            if (!threeCamera || !threeRenderer) return;
            const container = document.getElementById('preview-container');
            threeCamera.aspect = container.clientWidth / 500;
            threeCamera.updateProjectionMatrix();
            threeRenderer.setSize(container.clientWidth, 500);
        }

        function resetCamera() {
            if (threeCamera && threeControls) {
                threeCamera.position.set(2, 2, 2);
                threeControls.reset();
            }
        }

        let wireframeMode = false;
        function toggleWireframe() {
            if (!threeScene) return;
            wireframeMode = !wireframeMode;
            threeScene.traverse((obj) => {
                if (obj.isMesh && obj.material) {
                    obj.material.wireframe = wireframeMode;
                }
            });
        }

        function toggleGrid() {
            if (!threeScene) return;
            const grid = threeScene.getObjectByName('grid');
            if (grid) grid.visible = !grid.visible;
        }

        function renderUnsupported(meta) {
            cleanup3D();
            const container = document.getElementById('preview-container');
            container.className = 'preview-container';
            container.innerHTML = `
                <div class="placeholder">
                    <h2>Preview Not Available</h2>
                    <p>Type: ${meta.content_type}</p>
                    <p>Size: ${formatSize(meta.size_bytes)}</p>
                    <p><a href="/api/preview/artifact/${meta.hash}/raw" style="color:#e94560;">Download file</a></p>
                </div>
            `;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>"""
