# Installation & Setup Guide

This guide covers installing and configuring the Transformation Portal for development, production, and RAG (Retrieval-Augmented Generation) capabilities.

## Prerequisites

Before installing, ensure you have the following:

### Required

- **Python 3.8+** (3.10+ recommended for best compatibility)
  - Verify with: `python3 --version`
- **Git** for version control
  - Verify with: `git --version`
- **pip** (usually included with Python)
  - Verify with: `pip --version`

### Optional (for ML/AI Features)

- **CUDA-capable GPU** (NVIDIA) for accelerated ML processing
  - CUDA 11.8+ recommended for PyTorch compatibility
- **Apple Silicon** (M1, M2, M3 series) with Metal Performance Shaders (MPS)
  - No additional setup required on macOS
- **FFmpeg 6+** for video processing features
  - Install: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

## Quick Start

The fastest way to get started is using the installation script:

```bash
# Clone the repository (if not already done)
git clone https://github.com/RC219805/Transformation_Portal.git
cd Transformation_Portal

# Run the installation script
./install.sh

# Activate the virtual environment
source venv/bin/activate

# Verify installation
make test-fast
```

The installation script will:
1. Check Python version (3.8+ required)
2. Create a virtual environment (`venv/`)
3. Install dependencies from `requirements.txt`
4. Install development dependencies from `requirements-dev.txt` (if present)
5. Create the required directory structure for the RAG system
6. Set up the environment configuration (`.env`)
7. Make shell scripts executable

## Manual Installation

If you prefer manual installation or need to customize the process:

### Step 1: Create Virtual Environment

```bash
# Navigate to the repository
cd Transformation_Portal

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional but recommended)
pip install -r requirements-dev.txt
```

For full ML capabilities (PyTorch, Diffusers, ControlNet):
```bash
pip install -e ".[ml]"
```

### Step 3: Create Directory Structure

The RAG system requires specific directories. Create them manually:

```bash
mkdir -p data/knowledge_base/memory_snapshots
mkdir -p data/feedback_loops/audits
mkdir -p assets/luts/imported
mkdir -p assets/models
mkdir -p src/transformation_portal/pipelines
mkdir -p scripts/production
mkdir -p scripts/utilities
mkdir -p docs/guides
mkdir -p archive/debug_artifacts
```

### Step 4: Environment Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

### Step 5: Make Scripts Executable

```bash
chmod +x install.sh
chmod +x scripts/*.sh
find scripts/ -name "*.sh" -exec chmod +x {} \;
```

## RAG Configuration

The Transformation Portal includes a RAG (Retrieval-Augmented Generation) system for AI-enhanced processing workflows.

### Knowledge Base Structure

The RAG system uses the following directory structure:

```
data/
└── knowledge_base/
    └── memory_snapshots/    # Stored context and memory
```

This structure is created automatically by `install.sh`.

### Model Weights

For AI features, you may need to place model weights in `assets/models/`:

- **Depth Estimation**: Depth Anything V2 models (downloaded automatically on first use)
- **Upscaling**: Real-ESRGAN weights (optional, for resolution enhancement)
- **ControlNet**: Various ControlNet models for guided generation

Most models are downloaded automatically when first used. For offline usage, pre-download models:

```bash
python scripts/download_depth_models.py
```

### Environment Variables

Key RAG-related environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSFORMATION_ENV` | Environment mode | `development` |
| `RAG_MEMORY_PATH` | Path to knowledge base | `data/knowledge_base` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## Verifying the Installation

### Run Tests

```bash
# Quick test (recommended first check)
make test-fast

# Full test suite
make test-full

# Run specific test file
pytest tests/test_depth_tools.py -v
```

### Check Directory Structure

```bash
# Verify key directories exist
ls -la data/knowledge_base/
ls -la assets/models/
ls -la scripts/utilities/
```

### Run Linting

```bash
make lint
```

### Verify Python Environment

```bash
# Should show packages in venv
pip list

# Check package is installed
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

## Troubleshooting

### Common Issues

**"Python not found"**
- Ensure Python 3.8+ is installed and in your PATH
- Try `python3` instead of `python`

**Import errors after installation**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**Permission denied on scripts**
- Run: `chmod +x install.sh scripts/*.sh`

**ML features not working**
- Install ML extras: `pip install -e ".[ml]"`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Getting Help

- Check `docs/TROUBLESHOOTING.md` for detailed solutions
- Run `make help` to see available make targets
- Review existing issues on GitHub

## Next Steps

After installation:

1. **Explore the documentation**: See `docs/README.md` for an overview
2. **Run example pipelines**: Check `examples/` directory
3. **Configure presets**: Review `config/` for processing presets
4. **Read the architecture guide**: See `docs/ARCHITECTURE.md`
