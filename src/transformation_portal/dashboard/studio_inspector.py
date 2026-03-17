"""Studio-grade 3D inspector for artifact visualization.

This module provides a professional 3D inspection environment with:
- HDR environment lighting (IBL) with environment selector
- Material inspector (metalness, roughness, texture preview)
- Multi-view comparison (side-by-side)
- Depth visualization mode
- Segmentation overlay support
- GPU picking (click to highlight mesh)
- Advanced camera controls
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Optional FastAPI import
try:
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None


def create_studio_inspector_router() -> "APIRouter":
    """Create the studio inspector router.

    Returns:
        FastAPI APIRouter with studio inspector endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required for studio inspector")

    router = APIRouter(prefix="/api/studio", tags=["studio"])

    @router.get("/", response_class=HTMLResponse)
    async def studio_inspector():
        """Serve the studio-grade 3D inspector."""
        return get_studio_inspector_html()

    @router.get("/compare", response_class=HTMLResponse)
    async def comparison_view():
        """Serve the multi-view comparison interface."""
        return get_comparison_view_html()

    return router


def get_studio_inspector_html() -> str:
    """Get the studio-grade 3D inspector HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Studio 3D Inspector</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/RGBELoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/OBJLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/PLYLoader.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            background: #16213e;
            padding: 0.75rem 1.5rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
        }
        .header h1 { font-size: 1rem; }
        .header-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .header-controls input {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.4rem 0.75rem;
            border-radius: 0.25rem;
            width: 350px;
            font-family: monospace;
            font-size: 0.8rem;
        }
        .header-controls button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.4rem 0.75rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .main {
            display: flex;
            height: calc(100vh - 50px);
        }
        .sidebar {
            width: 280px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            overflow-y: auto;
            flex-shrink: 0;
        }
        .panel {
            border-bottom: 1px solid #0f3460;
        }
        .panel-header {
            padding: 0.75rem 1rem;
            background: #0f3460;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .panel-header:hover { background: #1a3a6e; }
        .panel-content {
            padding: 1rem;
            display: none;
        }
        .panel.open .panel-content { display: block; }
        .panel-row {
            margin-bottom: 0.75rem;
        }
        .panel-row:last-child { margin-bottom: 0; }
        .panel-row label {
            display: block;
            font-size: 0.7rem;
            color: #94a3b8;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }
        .panel-row select, .panel-row input[type="range"] {
            width: 100%;
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.4rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
        }
        .panel-row input[type="range"] {
            padding: 0;
            height: 6px;
            -webkit-appearance: none;
            background: #0f3460;
        }
        .panel-row input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #e94560;
            cursor: pointer;
        }
        .value-display {
            font-size: 0.75rem;
            color: #e94560;
            float: right;
        }
        .btn-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .btn-row button {
            flex: 1;
            min-width: 60px;
            background: #0f3460;
            border: none;
            color: #eee;
            padding: 0.4rem 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.7rem;
        }
        .btn-row button:hover { background: #e94560; }
        .btn-row button.active { background: #e94560; }
        .viewport {
            flex: 1;
            position: relative;
            background: #0a0a15;
        }
        #canvas-container {
            width: 100%;
            height: 100%;
        }
        #canvas-container canvas {
            width: 100% !important;
            height: 100% !important;
        }
        .viewport-overlay {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .viewport-overlay button {
            background: rgba(15, 52, 96, 0.9);
            border: none;
            color: #eee;
            padding: 0.5rem 0.75rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.75rem;
            backdrop-filter: blur(4px);
        }
        .viewport-overlay button:hover { background: rgba(233, 69, 96, 0.9); }
        .viewport-overlay button.active { background: rgba(233, 69, 96, 0.9); }
        .info-bar {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(22, 33, 62, 0.95);
            padding: 0.5rem 1rem;
            font-size: 0.75rem;
            display: flex;
            justify-content: space-between;
            border-top: 1px solid #0f3460;
        }
        .info-bar .stat { color: #94a3b8; }
        .info-bar .value { color: #eee; margin-left: 0.5rem; }
        .picked-info {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(22, 33, 62, 0.95);
            padding: 0.75rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            display: none;
            max-width: 250px;
        }
        .picked-info.visible { display: block; }
        .picked-info h4 { margin-bottom: 0.5rem; color: #e94560; }
        .picked-info .prop { color: #94a3b8; margin-bottom: 0.25rem; }
        .color-preview {
            display: inline-block;
            width: 16px;
            height: 16px;
            border-radius: 2px;
            vertical-align: middle;
            margin-left: 0.5rem;
            border: 1px solid #0f3460;
        }
        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(26, 26, 46, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 1rem;
            z-index: 10;
        }
        .loading-overlay.hidden { display: none; }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #0f3460;
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Studio 3D Inspector</h1>
        <div class="header-controls">
            <input type="text" id="hash-input" placeholder="Enter artifact hash or URL...">
            <button onclick="loadModel()">Load</button>
            <button onclick="window.location.href='/api/studio/compare'">Compare View</button>
        </div>
    </div>
    <div class="main">
        <div class="sidebar">
            <!-- Environment Panel -->
            <div class="panel open">
                <div class="panel-header" onclick="togglePanel(this)">
                    <span>Environment</span>
                    <span>▼</span>
                </div>
                <div class="panel-content">
                    <div class="panel-row">
                        <label>HDR Environment</label>
                        <select id="env-select" onchange="loadEnvironment(this.value)">
                            <option value="royal_esplanade_1k.hdr">Studio</option>
                            <option value="venice_sunset_1k.hdr">Sunset</option>
                            <option value="lebombo_1k.hdr">Neutral</option>
                            <option value="moonless_golf_1k.hdr">Night</option>
                            <option value="industrial_sunset_puresky_1k.hdr">Industrial</option>
                            <option value="none">None (Solid)</option>
                        </select>
                    </div>
                    <div class="panel-row">
                        <label>Background</label>
                        <div class="btn-row">
                            <button onclick="setBackground('env')" class="active" id="bg-env">HDR</button>
                            <button onclick="setBackground('solid')" id="bg-solid">Solid</button>
                            <button onclick="setBackground('gradient')" id="bg-gradient">Gradient</button>
                        </div>
                    </div>
                    <div class="panel-row">
                        <label>Exposure <span class="value-display" id="exposure-val">1.0</span></label>
                        <input type="range" min="0.1" max="3" step="0.1" value="1" onchange="setExposure(this.value)">
                    </div>
                </div>
            </div>

            <!-- Lighting Panel -->
            <div class="panel open">
                <div class="panel-header" onclick="togglePanel(this)">
                    <span>Lighting</span>
                    <span>▼</span>
                </div>
                <div class="panel-content">
                    <div class="panel-row">
                        <label>Preset</label>
                        <div class="btn-row">
                            <button onclick="setLightingPreset('studio')" class="active">Studio</button>
                            <button onclick="setLightingPreset('dramatic')">Dramatic</button>
                            <button onclick="setLightingPreset('flat')">Flat</button>
                        </div>
                    </div>
                    <div class="panel-row">
                        <label>Ambient <span class="value-display" id="ambient-val">0.5</span></label>
                        <input type="range" min="0" max="2" step="0.1" value="0.5" onchange="setAmbient(this.value)">
                    </div>
                    <div class="panel-row">
                        <label>Key Light <span class="value-display" id="key-val">1.5</span></label>
                        <input type="range" min="0" max="5" step="0.1" value="1.5" onchange="setKeyLight(this.value)">
                    </div>
                </div>
            </div>

            <!-- Material Panel -->
            <div class="panel open">
                <div class="panel-header" onclick="togglePanel(this)">
                    <span>Material Override</span>
                    <span>▼</span>
                </div>
                <div class="panel-content">
                    <div class="panel-row">
                        <label>Metalness <span class="value-display" id="metal-val">—</span></label>
                        <input type="range" min="0" max="1" step="0.01" value="0.5" onchange="setMetalness(this.value)">
                    </div>
                    <div class="panel-row">
                        <label>Roughness <span class="value-display" id="rough-val">—</span></label>
                        <input type="range" min="0" max="1" step="0.01" value="0.5" onchange="setRoughness(this.value)">
                    </div>
                    <div class="panel-row">
                        <label>Env Map Intensity <span class="value-display" id="envmap-val">1.0</span></label>
                        <input type="range" min="0" max="3" step="0.1" value="1" onchange="setEnvMapIntensity(this.value)">
                    </div>
                    <div class="panel-row">
                        <label>Override Color</label>
                        <input type="color" value="#e94560" onchange="setOverrideColor(this.value)" style="width:100%;height:30px;">
                    </div>
                </div>
            </div>

            <!-- View Mode Panel -->
            <div class="panel open">
                <div class="panel-header" onclick="togglePanel(this)">
                    <span>View Mode</span>
                    <span>▼</span>
                </div>
                <div class="panel-content">
                    <div class="panel-row">
                        <div class="btn-row">
                            <button onclick="setViewMode('normal')" class="active" id="mode-normal">Normal</button>
                            <button onclick="setViewMode('wireframe')" id="mode-wireframe">Wire</button>
                            <button onclick="setViewMode('depth')" id="mode-depth">Depth</button>
                        </div>
                    </div>
                    <div class="panel-row">
                        <div class="btn-row">
                            <button onclick="setViewMode('normals')" id="mode-normals">Normals</button>
                            <button onclick="setViewMode('uv')" id="mode-uv">UV</button>
                            <button onclick="setViewMode('ao')" id="mode-ao">AO</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Overlay Panel -->
            <div class="panel">
                <div class="panel-header" onclick="togglePanel(this)">
                    <span>Overlays</span>
                    <span>▼</span>
                </div>
                <div class="panel-content">
                    <div class="panel-row">
                        <label>Segmentation Mask</label>
                        <input type="text" id="seg-hash" placeholder="Mask artifact hash..." style="width:100%;margin-bottom:0.5rem;">
                        <button onclick="applySegmentationOverlay()" style="width:100%;">Apply Overlay</button>
                    </div>
                    <div class="panel-row">
                        <label>Depth Map</label>
                        <input type="text" id="depth-hash" placeholder="Depth artifact hash..." style="width:100%;margin-bottom:0.5rem;">
                        <button onclick="applyDepthOverlay()" style="width:100%;">Apply Depth</button>
                    </div>
                </div>
            </div>

            <!-- Display Panel -->
            <div class="panel">
                <div class="panel-header" onclick="togglePanel(this)">
                    <span>Display</span>
                    <span>▼</span>
                </div>
                <div class="panel-content">
                    <div class="panel-row">
                        <div class="btn-row">
                            <button onclick="toggleGrid()" id="btn-grid" class="active">Grid</button>
                            <button onclick="toggleAxes()" id="btn-axes">Axes</button>
                            <button onclick="toggleBounds()" id="btn-bounds">Bounds</button>
                        </div>
                    </div>
                    <div class="panel-row">
                        <div class="btn-row">
                            <button onclick="toggleAutoRotate()" id="btn-rotate">Auto Rotate</button>
                            <button onclick="resetCamera()">Reset View</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="viewport">
            <div id="canvas-container"></div>

            <div class="viewport-overlay">
                <button onclick="screenshotViewport()">📷 Screenshot</button>
                <button onclick="toggleFullscreen()">⛶ Fullscreen</button>
            </div>

            <div class="picked-info" id="picked-info">
                <h4>Selected Object</h4>
                <div id="picked-details"></div>
            </div>

            <div class="info-bar">
                <div>
                    <span class="stat">Vertices:</span><span class="value" id="stat-verts">0</span>
                    <span class="stat" style="margin-left:1rem;">Faces:</span><span class="value" id="stat-faces">0</span>
                    <span class="stat" style="margin-left:1rem;">Objects:</span><span class="value" id="stat-objects">0</span>
                </div>
                <div>
                    <span class="stat">FPS:</span><span class="value" id="stat-fps">0</span>
                </div>
            </div>

            <div class="loading-overlay" id="loading-overlay">
                <div class="spinner"></div>
                <div>Loading model...</div>
            </div>
        </div>
    </div>

    <script>
        // ============================================================
        // STATE
        // ============================================================
        const state = {
            scene: null,
            camera: null,
            renderer: null,
            controls: null,
            meshes: [],
            originalMaterials: new Map(),
            model: null,
            grid: null,
            axes: null,
            boundingBox: null,
            ambientLight: null,
            keyLight: null,
            fillLight: null,
            currentEnv: null,
            viewMode: 'normal',
            picking: true,
        };

        let frameCount = 0;
        let lastTime = performance.now();

        // ============================================================
        // INITIALIZATION
        // ============================================================
        function init() {
            const container = document.getElementById('canvas-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            // Scene
            state.scene = new THREE.Scene();

            // Renderer
            state.renderer = new THREE.WebGLRenderer({
                antialias: true,
                preserveDrawingBuffer: true,
            });
            state.renderer.setSize(width, height);
            state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            state.renderer.outputEncoding = THREE.sRGBEncoding;
            state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            state.renderer.toneMappingExposure = 1.0;
            state.renderer.shadowMap.enabled = true;
            state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(state.renderer.domElement);

            // Camera
            state.camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
            state.camera.position.set(3, 2, 3);

            // Controls
            state.controls = new THREE.OrbitControls(state.camera, state.renderer.domElement);
            state.controls.enableDamping = true;
            state.controls.dampingFactor = 0.05;
            state.controls.screenSpacePanning = true;
            state.controls.maxDistance = 50;
            state.controls.minDistance = 0.1;

            // Lighting
            setupLighting();

            // Grid
            state.grid = new THREE.GridHelper(10, 20, 0x0f3460, 0x0f3460);
            state.scene.add(state.grid);

            // Axes
            state.axes = new THREE.AxesHelper(2);
            state.axes.visible = false;
            state.scene.add(state.axes);

            // Load default environment
            loadEnvironment('royal_esplanade_1k.hdr');

            // Raycaster for picking
            setupPicking();

            // Window resize
            window.addEventListener('resize', onResize);

            // Start render loop
            animate();

            // Hide loading
            hideLoading();

            // Check URL params
            const params = new URLSearchParams(window.location.search);
            if (params.get('hash')) {
                document.getElementById('hash-input').value = params.get('hash');
                loadModel();
            }
        }

        function setupLighting() {
            // Ambient
            state.ambientLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.5);
            state.scene.add(state.ambientLight);

            // Key light
            state.keyLight = new THREE.DirectionalLight(0xffffff, 1.5);
            state.keyLight.position.set(5, 10, 7.5);
            state.keyLight.castShadow = true;
            state.keyLight.shadow.mapSize.width = 2048;
            state.keyLight.shadow.mapSize.height = 2048;
            state.scene.add(state.keyLight);

            // Fill light
            state.fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
            state.fillLight.position.set(-5, 5, -5);
            state.scene.add(state.fillLight);
        }

        function setupPicking() {
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            state.renderer.domElement.addEventListener('click', (event) => {
                if (!state.picking || state.meshes.length === 0) return;

                const rect = state.renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                raycaster.setFromCamera(mouse, state.camera);
                const intersects = raycaster.intersectObjects(state.meshes, true);

                // Reset previous selection
                state.meshes.forEach(m => {
                    if (m.material && m.material.emissive) {
                        m.material.emissive.setHex(0x000000);
                    }
                });

                if (intersects.length > 0) {
                    const obj = intersects[0].object;
                    if (obj.material && obj.material.emissive) {
                        obj.material.emissive.setHex(0x331111);
                    }
                    showPickedInfo(obj, intersects[0]);
                } else {
                    hidePickedInfo();
                }
            });
        }

        // ============================================================
        // MODEL LOADING
        // ============================================================
        function loadModel() {
            const input = document.getElementById('hash-input').value.trim();
            if (!input) return;

            showLoading();

            // Determine URL
            let url;
            if (input.startsWith('http')) {
                url = input;
            } else {
                url = `/api/preview/artifact/${input}/raw`;
            }

            // Clear previous model
            if (state.model) {
                state.scene.remove(state.model);
                state.model = null;
            }
            state.meshes = [];
            state.originalMaterials.clear();

            // Detect format and load
            const loader = new THREE.GLTFLoader();

            loader.load(
                url,
                (gltf) => {
                    state.model = gltf.scene;
                    processModel(state.model);
                    state.scene.add(state.model);
                    fitCameraToModel();
                    updateStats();
                    hideLoading();
                },
                (progress) => {
                    // Progress callback
                },
                (error) => {
                    console.error('Load error:', error);
                    hideLoading();
                    alert('Failed to load model. Check the hash or URL.');
                }
            );
        }

        function processModel(model) {
            let vertCount = 0;
            let faceCount = 0;

            model.traverse((child) => {
                if (child.isMesh) {
                    state.meshes.push(child);

                    // Store original material
                    state.originalMaterials.set(child.uuid, child.material.clone());

                    // Ensure PBR properties
                    if (child.material) {
                        child.material.envMapIntensity = 1.0;
                        child.material.needsUpdate = true;
                    }

                    // Count geometry
                    if (child.geometry) {
                        const pos = child.geometry.attributes.position;
                        if (pos) vertCount += pos.count;
                        if (child.geometry.index) {
                            faceCount += child.geometry.index.count / 3;
                        } else if (pos) {
                            faceCount += pos.count / 3;
                        }
                    }

                    // Enable shadows
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            document.getElementById('stat-verts').textContent = vertCount.toLocaleString();
            document.getElementById('stat-faces').textContent = faceCount.toLocaleString();
            document.getElementById('stat-objects').textContent = state.meshes.length;
        }

        function fitCameraToModel() {
            if (!state.model) return;

            const box = new THREE.Box3().setFromObject(state.model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            // Center model at origin
            state.model.position.sub(center);

            // Normalize scale
            const maxDim = Math.max(size.x, size.y, size.z);
            if (maxDim > 0) {
                const scale = 2 / maxDim;
                state.model.scale.setScalar(scale);
            }

            // Position camera
            state.camera.position.set(2, 1.5, 2);
            state.controls.target.set(0, 0, 0);
            state.controls.update();
        }

        // ============================================================
        // ENVIRONMENT
        // ============================================================
        function loadEnvironment(name) {
            if (name === 'none') {
                state.scene.environment = null;
                state.scene.background = new THREE.Color(0x1a1a2e);
                return;
            }

            const loader = new THREE.RGBELoader();
            const url = `https://cdn.jsdelivr.net/gh/mrdoob/three.js@r158/examples/textures/equirectangular/${name}`;

            loader.load(url, (texture) => {
                texture.mapping = THREE.EquirectangularReflectionMapping;
                state.scene.environment = texture;
                state.currentEnv = texture;

                const bgBtn = document.getElementById('bg-env');
                if (bgBtn && bgBtn.classList.contains('active')) {
                    state.scene.background = texture;
                }
            });
        }

        function setBackground(mode) {
            document.querySelectorAll('#bg-env, #bg-solid, #bg-gradient').forEach(b => b.classList.remove('active'));

            if (mode === 'env' && state.currentEnv) {
                state.scene.background = state.currentEnv;
                document.getElementById('bg-env').classList.add('active');
            } else if (mode === 'solid') {
                state.scene.background = new THREE.Color(0x1a1a2e);
                document.getElementById('bg-solid').classList.add('active');
            } else if (mode === 'gradient') {
                // Create gradient texture
                const canvas = document.createElement('canvas');
                canvas.width = 2;
                canvas.height = 512;
                const ctx = canvas.getContext('2d');
                const gradient = ctx.createLinearGradient(0, 0, 0, 512);
                gradient.addColorStop(0, '#1a1a2e');
                gradient.addColorStop(1, '#0f3460');
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, 2, 512);
                const tex = new THREE.CanvasTexture(canvas);
                state.scene.background = tex;
                document.getElementById('bg-gradient').classList.add('active');
            }
        }

        function setExposure(val) {
            state.renderer.toneMappingExposure = parseFloat(val);
            document.getElementById('exposure-val').textContent = parseFloat(val).toFixed(1);
        }

        // ============================================================
        // LIGHTING
        // ============================================================
        function setLightingPreset(preset) {
            document.querySelectorAll('.panel-content .btn-row button').forEach(b => {
                if (b.textContent.toLowerCase().includes(preset.toLowerCase())) {
                    b.classList.add('active');
                }
            });

            switch (preset) {
                case 'studio':
                    state.ambientLight.intensity = 0.5;
                    state.keyLight.intensity = 1.5;
                    state.fillLight.intensity = 0.5;
                    break;
                case 'dramatic':
                    state.ambientLight.intensity = 0.2;
                    state.keyLight.intensity = 2.5;
                    state.fillLight.intensity = 0.2;
                    break;
                case 'flat':
                    state.ambientLight.intensity = 1.0;
                    state.keyLight.intensity = 0.5;
                    state.fillLight.intensity = 0.5;
                    break;
            }
        }

        function setAmbient(val) {
            state.ambientLight.intensity = parseFloat(val);
            document.getElementById('ambient-val').textContent = parseFloat(val).toFixed(1);
        }

        function setKeyLight(val) {
            state.keyLight.intensity = parseFloat(val);
            document.getElementById('key-val').textContent = parseFloat(val).toFixed(1);
        }

        // ============================================================
        // MATERIAL
        // ============================================================
        function setMetalness(val) {
            const v = parseFloat(val);
            document.getElementById('metal-val').textContent = v.toFixed(2);
            state.meshes.forEach(m => {
                if (m.material && m.material.metalness !== undefined) {
                    m.material.metalness = v;
                    m.material.needsUpdate = true;
                }
            });
        }

        function setRoughness(val) {
            const v = parseFloat(val);
            document.getElementById('rough-val').textContent = v.toFixed(2);
            state.meshes.forEach(m => {
                if (m.material && m.material.roughness !== undefined) {
                    m.material.roughness = v;
                    m.material.needsUpdate = true;
                }
            });
        }

        function setEnvMapIntensity(val) {
            const v = parseFloat(val);
            document.getElementById('envmap-val').textContent = v.toFixed(1);
            state.meshes.forEach(m => {
                if (m.material) {
                    m.material.envMapIntensity = v;
                    m.material.needsUpdate = true;
                }
            });
        }

        function setOverrideColor(hex) {
            const color = new THREE.Color(hex);
            state.meshes.forEach(m => {
                if (m.material && m.material.color) {
                    m.material.color = color;
                    m.material.needsUpdate = true;
                }
            });
        }

        // ============================================================
        // VIEW MODES
        // ============================================================
        function setViewMode(mode) {
            state.viewMode = mode;

            // Update buttons
            document.querySelectorAll('[id^="mode-"]').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById('mode-' + mode);
            if (btn) btn.classList.add('active');

            state.meshes.forEach(mesh => {
                const original = state.originalMaterials.get(mesh.uuid);

                switch (mode) {
                    case 'normal':
                        if (original) mesh.material = original.clone();
                        mesh.material.wireframe = false;
                        break;
                    case 'wireframe':
                        if (original) mesh.material = original.clone();
                        mesh.material.wireframe = true;
                        break;
                    case 'depth':
                        mesh.material = new THREE.MeshDepthMaterial();
                        break;
                    case 'normals':
                        mesh.material = new THREE.MeshNormalMaterial();
                        break;
                    case 'uv':
                        mesh.material = new THREE.MeshBasicMaterial({
                            map: createUVTexture(),
                        });
                        break;
                    case 'ao':
                        mesh.material = new THREE.MeshBasicMaterial({
                            color: 0xffffff,
                            aoMapIntensity: 1.0,
                        });
                        break;
                }
                mesh.material.needsUpdate = true;
            });
        }

        function createUVTexture() {
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');

            // Checker pattern
            const size = 32;
            for (let y = 0; y < 256; y += size) {
                for (let x = 0; x < 256; x += size) {
                    ctx.fillStyle = ((x + y) / size) % 2 === 0 ? '#e94560' : '#0f3460';
                    ctx.fillRect(x, y, size, size);
                }
            }

            return new THREE.CanvasTexture(canvas);
        }

        // ============================================================
        // OVERLAYS
        // ============================================================
        function applySegmentationOverlay() {
            const hash = document.getElementById('seg-hash').value.trim();
            if (!hash) return;

            const loader = new THREE.TextureLoader();
            loader.load(`/api/preview/artifact/${hash}/raw`, (texture) => {
                state.meshes.forEach(m => {
                    if (m.material) {
                        m.material.emissiveMap = texture;
                        m.material.emissive = new THREE.Color(0xffffff);
                        m.material.emissiveIntensity = 0.5;
                        m.material.needsUpdate = true;
                    }
                });
            });
        }

        function applyDepthOverlay() {
            const hash = document.getElementById('depth-hash').value.trim();
            if (!hash) return;

            const loader = new THREE.TextureLoader();
            loader.load(`/api/preview/artifact/${hash}/raw`, (texture) => {
                state.meshes.forEach(m => {
                    m.material = new THREE.MeshBasicMaterial({
                        map: texture,
                    });
                    m.material.needsUpdate = true;
                });
            });
        }

        // ============================================================
        // DISPLAY TOGGLES
        // ============================================================
        function toggleGrid() {
            state.grid.visible = !state.grid.visible;
            document.getElementById('btn-grid').classList.toggle('active', state.grid.visible);
        }

        function toggleAxes() {
            state.axes.visible = !state.axes.visible;
            document.getElementById('btn-axes').classList.toggle('active', state.axes.visible);
        }

        function toggleBounds() {
            if (state.boundingBox) {
                state.scene.remove(state.boundingBox);
                state.boundingBox = null;
                document.getElementById('btn-bounds').classList.remove('active');
            } else if (state.model) {
                const box = new THREE.Box3().setFromObject(state.model);
                state.boundingBox = new THREE.Box3Helper(box, 0xe94560);
                state.scene.add(state.boundingBox);
                document.getElementById('btn-bounds').classList.add('active');
            }
        }

        function toggleAutoRotate() {
            state.controls.autoRotate = !state.controls.autoRotate;
            state.controls.autoRotateSpeed = 1.0;
            document.getElementById('btn-rotate').classList.toggle('active', state.controls.autoRotate);
        }

        function resetCamera() {
            state.camera.position.set(2, 1.5, 2);
            state.controls.target.set(0, 0, 0);
            state.controls.reset();
        }

        // ============================================================
        // PICKING INFO
        // ============================================================
        function showPickedInfo(obj, intersection) {
            const panel = document.getElementById('picked-info');
            const details = document.getElementById('picked-details');

            let html = `<div class="prop">Name: ${obj.name || 'Unnamed'}</div>`;

            if (obj.geometry) {
                const verts = obj.geometry.attributes.position?.count || 0;
                html += `<div class="prop">Vertices: ${verts.toLocaleString()}</div>`;
            }

            if (obj.material) {
                if (obj.material.color) {
                    const c = obj.material.color.getHexString();
                    html += `<div class="prop">Color: #${c} <span class="color-preview" style="background:#${c}"></span></div>`;
                }
                if (obj.material.metalness !== undefined) {
                    html += `<div class="prop">Metalness: ${obj.material.metalness.toFixed(2)}</div>`;
                }
                if (obj.material.roughness !== undefined) {
                    html += `<div class="prop">Roughness: ${obj.material.roughness.toFixed(2)}</div>`;
                }
            }

            html += `<div class="prop">Distance: ${intersection.distance.toFixed(2)}</div>`;

            details.innerHTML = html;
            panel.classList.add('visible');
        }

        function hidePickedInfo() {
            document.getElementById('picked-info').classList.remove('visible');
        }

        // ============================================================
        // UTILITY
        // ============================================================
        function togglePanel(header) {
            header.parentElement.classList.toggle('open');
        }

        function showLoading() {
            document.getElementById('loading-overlay').classList.remove('hidden');
        }

        function hideLoading() {
            document.getElementById('loading-overlay').classList.add('hidden');
        }

        function onResize() {
            const container = document.getElementById('canvas-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            state.camera.aspect = width / height;
            state.camera.updateProjectionMatrix();
            state.renderer.setSize(width, height);
        }

        function updateStats() {
            // FPS counter
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                document.getElementById('stat-fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
            }
        }

        function screenshotViewport() {
            const dataUrl = state.renderer.domElement.toDataURL('image/png');
            const link = document.createElement('a');
            link.download = 'viewport-screenshot.png';
            link.href = dataUrl;
            link.click();
        }

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.querySelector('.viewport').requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        }

        // ============================================================
        // RENDER LOOP
        // ============================================================
        function animate() {
            requestAnimationFrame(animate);
            state.controls.update();
            state.renderer.render(state.scene, state.camera);
            updateStats();
        }

        // Initialize on load
        window.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>"""


def get_comparison_view_html() -> str:
    """Get the multi-view comparison HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/GLTFLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/loaders/RGBELoader.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            background: #16213e;
            padding: 0.75rem 1.5rem;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
        }
        .header h1 { font-size: 1rem; }
        .header-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .input-group {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        .input-group label {
            font-size: 0.75rem;
            color: #94a3b8;
        }
        .input-group input {
            background: #1a1a2e;
            border: 1px solid #0f3460;
            color: #eee;
            padding: 0.4rem 0.75rem;
            border-radius: 0.25rem;
            width: 200px;
            font-family: monospace;
            font-size: 0.8rem;
        }
        .header-controls button {
            background: #e94560;
            border: none;
            color: #fff;
            padding: 0.4rem 0.75rem;
            border-radius: 0.25rem;
            cursor: pointer;
            font-size: 0.8rem;
        }
        .comparison-container {
            display: flex;
            height: calc(100vh - 50px);
        }
        .viewport {
            flex: 1;
            position: relative;
            border-right: 1px solid #0f3460;
        }
        .viewport:last-child { border-right: none; }
        .viewport-label {
            position: absolute;
            top: 1rem;
            left: 1rem;
            background: rgba(22, 33, 62, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            z-index: 10;
        }
        .viewport canvas {
            width: 100% !important;
            height: 100% !important;
        }
        .sync-indicator {
            position: fixed;
            bottom: 1rem;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(233, 69, 96, 0.9);
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Comparison</h1>
        <div class="header-controls">
            <div class="input-group">
                <label>Left:</label>
                <input type="text" id="hash-left" placeholder="Artifact hash...">
            </div>
            <div class="input-group">
                <label>Right:</label>
                <input type="text" id="hash-right" placeholder="Artifact hash...">
            </div>
            <button onclick="loadComparison()">Compare</button>
            <button onclick="toggleSync()">Sync Views</button>
            <button onclick="window.location.href='/api/studio/'">Single View</button>
        </div>
    </div>
    <div class="comparison-container">
        <div class="viewport" id="viewport-left">
            <div class="viewport-label">Left Model</div>
        </div>
        <div class="viewport" id="viewport-right">
            <div class="viewport-label">Right Model</div>
        </div>
    </div>
    <div class="sync-indicator" id="sync-indicator" style="display:none;">Views Synchronized</div>

    <script>
        const viewers = {
            left: null,
            right: null
        };
        let syncEnabled = true;

        function createViewer(containerId) {
            const container = document.getElementById(containerId);
            const width = container.clientWidth;
            const height = container.clientHeight;

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            renderer.outputEncoding = THREE.sRGBEncoding;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            container.appendChild(renderer.domElement);

            const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 1000);
            camera.position.set(2, 1.5, 2);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // Lighting
            const ambient = new THREE.HemisphereLight(0xffffff, 0x444444, 0.5);
            scene.add(ambient);

            const key = new THREE.DirectionalLight(0xffffff, 1.5);
            key.position.set(5, 10, 7.5);
            scene.add(key);

            // Grid
            const grid = new THREE.GridHelper(10, 20, 0x0f3460, 0x0f3460);
            scene.add(grid);

            // Environment
            const rgbeLoader = new THREE.RGBELoader();
            rgbeLoader.load(
                'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r158/examples/textures/equirectangular/royal_esplanade_1k.hdr',
                (texture) => {
                    texture.mapping = THREE.EquirectangularReflectionMapping;
                    scene.environment = texture;
                }
            );

            return { scene, camera, renderer, controls, model: null };
        }

        function loadModelInto(viewer, hash) {
            if (!hash) return;

            const loader = new THREE.GLTFLoader();
            const url = `/api/preview/artifact/${hash}/raw`;

            loader.load(url, (gltf) => {
                if (viewer.model) {
                    viewer.scene.remove(viewer.model);
                }

                viewer.model = gltf.scene;

                // Center and scale
                const box = new THREE.Box3().setFromObject(viewer.model);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                viewer.model.position.sub(center);
                const scale = 2 / Math.max(size.x, size.y, size.z);
                viewer.model.scale.setScalar(scale);

                viewer.scene.add(viewer.model);
            });
        }

        function loadComparison() {
            const hashLeft = document.getElementById('hash-left').value.trim();
            const hashRight = document.getElementById('hash-right').value.trim();

            loadModelInto(viewers.left, hashLeft);
            loadModelInto(viewers.right, hashRight);
        }

        function toggleSync() {
            syncEnabled = !syncEnabled;
            document.getElementById('sync-indicator').style.display = syncEnabled ? 'block' : 'none';
        }

        function animate() {
            requestAnimationFrame(animate);

            if (syncEnabled && viewers.left && viewers.right) {
                // Sync camera positions
                viewers.right.camera.position.copy(viewers.left.camera.position);
                viewers.right.camera.rotation.copy(viewers.left.camera.rotation);
                viewers.right.controls.target.copy(viewers.left.controls.target);
            }

            if (viewers.left) {
                viewers.left.controls.update();
                viewers.left.renderer.render(viewers.left.scene, viewers.left.camera);
            }
            if (viewers.right) {
                viewers.right.controls.update();
                viewers.right.renderer.render(viewers.right.scene, viewers.right.camera);
            }
        }

        function onResize() {
            ['left', 'right'].forEach(side => {
                const container = document.getElementById(`viewport-${side}`);
                const viewer = viewers[side];
                if (viewer) {
                    const width = container.clientWidth;
                    const height = container.clientHeight;
                    viewer.camera.aspect = width / height;
                    viewer.camera.updateProjectionMatrix();
                    viewer.renderer.setSize(width, height);
                }
            });
        }

        // Initialize
        window.addEventListener('DOMContentLoaded', () => {
            viewers.left = createViewer('viewport-left');
            viewers.right = createViewer('viewport-right');
            animate();

            // Check URL params
            const params = new URLSearchParams(window.location.search);
            if (params.get('left')) {
                document.getElementById('hash-left').value = params.get('left');
            }
            if (params.get('right')) {
                document.getElementById('hash-right').value = params.get('right');
            }
            if (params.get('left') || params.get('right')) {
                loadComparison();
            }
        });

        window.addEventListener('resize', onResize);
    </script>
</body>
</html>"""
