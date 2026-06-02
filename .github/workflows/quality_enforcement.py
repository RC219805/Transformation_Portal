#!/usr/bin/env python3
"""
Quality Enforcement Script - Proactive Code Quality Standards
Runs before commits to catch common issues early.
"""
import subprocess
import sys
from pathlib import Path


def run_autopep8_fix():
    """Auto-fix formatting issues."""
    print("🔧 Running autopep8 auto-fixes...")
    cmd = [
        "find",
        ".",
        "-name",
        "*.py",
        "-type",
        "f",
        "!",
        "-path",
        "./deprecated/*",
        "!",
        "-path",
        "./src/transformation_portal/*",
        "!",
        "-path",
        "./.venv/*",
        "-exec",
        "autopep8",
        "--in-place",
        "--max-line-length=127",
        "{}",
        ";",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"❌ autopep8 failed: {result.stderr}")
        return False
    print("✅ Formatting fixes applied")
    return True


def check_critical_errors():
    """Check for critical flake8 errors only."""
    print("\n🔍 Checking for critical errors (undefined names, syntax errors)...")
    cmd = [
        "flake8",
        ".",
        "--count",
        "--select=E9,F63,F7,F82",  # Critical errors only
        "--show-source",
        "--statistics",
        "--exclude=deprecated/,src/transformation_portal/,.venv/,.backup_local/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"❌ Critical errors found:\n{result.stdout}")
        return False
    print("✅ No critical errors")
    return True


def check_imports():
    """Check for import-related issues."""
    print("\n📦 Checking imports...")
    cmd = [
        "flake8",
        ".",
        "--select=F401,F811",  # Unused imports, redefined names
        "--exclude=deprecated/,src/transformation_portal/,.venv/,.backup_local/,tests/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f"⚠️  Import issues found (non-blocking):\n{result.stdout}")
    else:
        print("✅ Imports clean")
    return True  # Non-blocking


def check_dataclass_errors():
    """Check for common dataclass issues."""
    print("\n🏗️  Checking dataclass definitions...")
    issues = []

    for py_file in Path(".").rglob("*.py"):
        if any(excl in str(py_file) for excl in ["deprecated", "src/transformation_portal", ".venv", ".backup_local"]):
            continue

        try:
            with open(py_file, "r") as f:
                lines = f.readlines()
                in_dataclass = False
                for i, line in enumerate(lines, 1):
                    if "@dataclass" in line:
                        in_dataclass = True
                    elif in_dataclass and "class " in line:
                        # Check next 20 lines for parameter issues
                        for j in range(i, min(i + 20, len(lines))):
                            if "finish_type=" in lines[j] and "finish_type:" in lines[j]:
                                issues.append(f"{py_file}:{j+1} - Possible duplicate parameter")
                        in_dataclass = False
        except Exception as e:
            print(f"⚠️  Could not check {py_file}: {e}")

    if issues:
        print(f"⚠️  Potential dataclass issues:\n" + "\n".join(issues))
    else:
        print("✅ Dataclass definitions look good")
    return True


def check_trailing_whitespace():
    """Check for excessive trailing whitespace."""
    print("\n🧹 Checking for trailing whitespace...")
    issues = 0

    for py_file in Path(".").rglob("*.py"):
        if any(excl in str(py_file) for excl in ["deprecated", "src/transformation_portal", ".venv", ".backup_local"]):
            continue

        try:
            with open(py_file, "r") as f:
                lines = f.readlines()
                ws_lines = [i + 1 for i, line in enumerate(lines) if line.endswith(" \n") or line.endswith("\t\n")]
                if len(ws_lines) > 10:  # More than 10 lines with trailing whitespace
                    issues += 1
                    print(f"  {py_file}: {len(ws_lines)} lines with trailing whitespace")
        except Exception:
            pass

    if issues > 0:
        print(f"⚠️  {issues} files with excessive trailing whitespace (should be auto-fixed by autopep8)")
    else:
        print("✅ Trailing whitespace under control")
    return True


def check_markdown_count():
    """Check root-level markdown file count."""
    print("\n📄 Checking markdown file count in root...")
    md_files = list(Path(".").glob("*.md"))
    count = len(md_files)

    if count > 10:
        print(f"❌ Too many markdown files in root ({count}/10):")
        for f in md_files:
            print(f"  - {f}")
        print("\n💡 Move documentation to docs/ subdirectories")
        return False

    print(f"✅ Markdown count OK ({count}/10)")
    return True


def main():
    """Run all quality checks."""
    print("=" * 60)
    print("🎯 Quality Enforcement - Proactive Code Standards")
    print("=" * 60)

    checks = [
        ("Auto-fix formatting", run_autopep8_fix),
        ("Critical errors", check_critical_errors),
        ("Imports", check_imports),
        ("Dataclass definitions", check_dataclass_errors),
        ("Trailing whitespace", check_trailing_whitespace),
        ("Markdown count", check_markdown_count),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)

    blocking_failures = []
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed and name in ["Critical errors", "Markdown count"]:
            blocking_failures.append(name)

    if blocking_failures:
        print(f"\n❌ Blocking failures: {', '.join(blocking_failures)}")
        print("\n💡 Fix these issues before committing")
        return 1

    print("\n✅ All critical checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
