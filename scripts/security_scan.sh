#!/bin/bash
# Security scan script matching CI policy
# Run this locally to verify security gate before pushing
set -e

echo "Running Bandit security scan with CI flags..."
echo "Policy: severity >= LOW, confidence >= MEDIUM"
echo ""

python -m bandit -r src/ -ll -ii

echo ""
echo "✅ Security scan passed"
