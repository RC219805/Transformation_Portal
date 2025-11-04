# Changelog

## 2025-11-04

### Breaking Changes

- **Dependency Management Restructure**: Core ML/AI dependencies (torch, diffusers, controlnet-aux, realesrgan) have been moved from the optional `[ai]` extra back to the main dependencies list. These packages are now always installed with the base package.
  - **Migration**: If you were previously installing with `pip install transformation-portal[ai]`, you can now use `pip install transformation-portal` as AI dependencies are included by default.
  - **Backward Compatibility**: The `[ai]` extra still exists for backward compatibility but is deprecated and will be removed in a future version.
  - **Rationale**: This change ensures consistent behavior across all installation methods and simplifies the dependency management for the core AI-powered pipelines that are central to the package functionality.

## 2025-10-03

- Enhanced `.github/copilot-instructions.md` with best practice sections following GitHub Copilot coding agent guidelines:
  - Added Repository Structure section with visual directory tree
  - Added Getting Started section with prerequisites and setup instructions
  - Added Troubleshooting section with common issues and solutions
  - Added Additional Resources section with links to internal documentation
  - Added Code Examples section with practical snippets for common tasks

## 2025-07-02

- Added integrated comprehensive dataset for Picacho Lane project under Client Deliverables.

## 2025-07-03
- Reconciled README guidance so the table of contents, section anchors, and terminology match the merged tooling pull requests.

## 2025-07-04
- Standardized README anchors and terminology for the tooling sections, including consistent tone-mapping language and nested table-of-contents links.
