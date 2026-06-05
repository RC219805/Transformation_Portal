#!/bin/bash
set -e

echo "Creating optimized directory structure..."

# Create main directories
mkdir -p scripts/{pipelines,utilities,analysis,setup}
mkdir -p examples/{pipelines,rag,workflows}
mkdir -p outputs/{750_picacho,tests,archive}
mkdir -p archive/{experiments,deprecated,legacy}

# Create README files for organization
cat > scripts/README.md << 'README'
# Scripts Directory

Organized executable scripts for the Transformation Portal.

## Structure
- **pipelines/** - Pipeline execution scripts (process_*, run_*)
- **utilities/** - Utility scripts (convert_*, fix_*, verify_*)
- **analysis/** - Analysis and diagnostic scripts (analyze_*, diagnose_*)
- **setup/** - Setup and installation scripts (install_*, download_*)
README

cat > examples/README.md << 'README'
# Examples Directory

Example code demonstrating various features and workflows.

## Structure
- **pipelines/** - Pipeline usage examples
- **rag/** - RAG system examples
- **workflows/** - Complete workflow demonstrations
README

cat > outputs/README.md << 'README'
# Outputs Directory

Generated outputs from pipeline executions and tests.

⚠️ This directory is gitignored - contents are not committed to the repository.

## Structure
- **750_picacho/** - Project-specific outputs
- **tests/** - Test execution outputs
- **archive/** - Archived outputs to preserve
README

cat > archive/README.md << 'README'
# Archive Directory

Legacy, deprecated, and experimental code preserved for reference.

## Structure
- **experiments/** - Experimental features and research
- **deprecated/** - Old implementations replaced by newer versions
- **legacy/** - Historical code for backward compatibility
README

echo "✓ Directory structure created"
ls -la scripts/ examples/ outputs/ archive/
