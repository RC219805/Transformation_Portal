"""CLI utility that surfaces temporal contracts, philosophy violations, and color token drift."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from codebase_philosophy_auditor import CodebasePhilosophyAuditor, Violation


@dataclass
class ValidUntilRecord:
    """Representation of a ``valid_until`` decorator instance."""

    target: str
    deadline: date
    reason: str
    path: Path
    line: int

    @property
    def days_remaining(self) -> int:
        return (self.deadline - date.today()).days


@dataclass
class PrincipleSummary:
    """Aggregate information about a philosophy principle's violations."""

    principle: str
    count: int
    examples: List[str]


@dataclass
class ColorTokenUsage:
    """Records how a color token is used across the codebase."""

    token: str
    hex_value: str
    used_in: List[str]


@dataclass
class ColorTokenReport:
    """Summary of all color tokens and their usage status."""

    tokens: List[ColorTokenUsage]
    orphans: List[ColorTokenUsage]


def collect_valid_until_records(tests_root: Path) -> List[ValidUntilRecord]:
    """Discover and sort ``valid_until`` decorators within *tests_root*."""

    records: List[ValidUntilRecord] = []
    for path in sorted(tests_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for decorator in node.decorator_list:
                record = _valid_until_from_decorator(decorator, path)
                if record is not None:
                    records.append(
                        ValidUntilRecord(
                            target=node.name,
                            deadline=record[0],
                            reason=record[1],
                            path=path,
                            line=decorator.lineno,
                        )
                    )

    return sorted(records, key=lambda item: item.days_remaining)


def collect_outdated_valid_until_records(tests_root: Path) -> List[ValidUntilRecord]:
    """Return ``valid_until`` records whose deadlines have passed."""

    today = date.today()
    return [
        record
        for record in collect_valid_until_records(tests_root)
        if record.deadline < today
    ]


def _valid_until_from_decorator(  # pylint: disable=too-many-branches
    decorator: ast.AST, path: Path
) -> Optional[tuple[date, str]]:
    """Extract deadline and reason from a valid_until decorator AST node."""
    if not isinstance(decorator, ast.Call):
        return None

    func = decorator.func
    if isinstance(func, ast.Name):
        func_name = func.id
    elif isinstance(func, ast.Attribute):
        func_name = func.attr
    else:
        return None

    if func_name != "valid_until":
        return None

    if not decorator.args:
        return None

    try:
        deadline_value = ast.literal_eval(decorator.args[0])
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to evaluate valid_until deadline in {path}") from exc

    if not isinstance(deadline_value, str):
        raise ValueError(
            f"valid_until decorator in {path} must use a string ISO date literal"
        )

    reason_value: Optional[str] = None
    if decorator.keywords:
        for keyword in decorator.keywords:
            if keyword.arg == "reason":
                reason_value = ast.literal_eval(keyword.value)
                break

    if reason_value is None:
        raise ValueError(
            f"valid_until decorator in {path} is missing required 'reason' keyword"
        )

    if not isinstance(reason_value, str):
        raise ValueError(
            f"valid_until decorator in {path} must use a string reason literal"
        )

    deadline = date.fromisoformat(deadline_value)
    return deadline, reason_value


def collect_philosophy_violations(
    paths: Iterable[Path], *, auditor: Optional[CodebasePhilosophyAuditor] = None
) -> Dict[str, PrincipleSummary]:
    """Aggregate :class:`Violation` instances returned by *auditor*."""

    auditor = auditor or CodebasePhilosophyAuditor()
    summaries: Dict[str, PrincipleSummary] = {}

    for module_path in _iter_python_files(paths):
        violations = auditor.audit_module(module_path)
        for violation in violations:
            summary = summaries.get(violation.principle)
            location = _format_violation_location(module_path, violation)
            if summary is None:
                summaries[violation.principle] = PrincipleSummary(
                    principle=violation.principle,
                    count=1,
                    examples=[location],
                )
            else:
                summary.count += 1
                if len(summary.examples) < 3:
                    summary.examples.append(location)

    return summaries


def _iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from (p for p in path.rglob("*.py") if p.is_file())
        elif path.suffix == ".py":
            yield path


def _format_violation_location(module_path: Path, violation: Violation) -> str:
    location = f"{module_path}" if violation.line is None else f"{module_path}:{violation.line}"
    return f"{location} – {violation.message}"


def collect_color_token_report(tokens_path: Path) -> ColorTokenReport:  # pylint: disable=too-many-locals
    """Return usage information for brand color tokens defined in *tokens_path*."""

    tokens_data = json.loads(tokens_path.read_text())
    brand_tokens = (
        tokens_data.get("tokens", {})
        .get("color", {})
        .get("brand", {})
    )

    directory = tokens_path.parent
    deliverables = [
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".css", ".js", ".mjs", ".cjs"}
    ]

    usages: List[ColorTokenUsage] = []
    orphans: List[ColorTokenUsage] = []

    for token_name, descriptor in sorted(brand_tokens.items(), key=lambda x: str(x[0])):
        value = descriptor.get("value")
        if not isinstance(value, str):
            continue
        normalized_hex = value.strip().lower()
        token_ref = f"color.brand.{token_name}"
        css_var = f"--brand-{token_name.replace('_', '-')}"

        used_in: List[str] = []
        for deliverable in deliverables:
            text = deliverable.read_text().lower()
            if (
                normalized_hex in text
                or token_ref in text
                or css_var in text
            ):
                used_in.append(deliverable.name)

        usage = ColorTokenUsage(
            token=token_name,
            hex_value=normalized_hex,
            used_in=used_in,
        )
        usages.append(usage)
        if not used_in:
            orphans.append(usage)

    return ColorTokenReport(tokens=usages, orphans=orphans)


def render_dashboard(
    valid_until_records: Sequence[ValidUntilRecord],
    principle_summaries: Dict[str, PrincipleSummary],
    color_report: ColorTokenReport,
) -> None:
    """Render a textual dashboard summarising the collected insights."""

    try:
        from rich.console import Console  # pylint: disable=import-outside-toplevel
        from rich.table import Table  # pylint: disable=import-outside-toplevel
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when Rich unavailable
        _render_plain_dashboard(valid_until_records, principle_summaries, color_report)
        return

    console = Console()

    console.rule("Temporal Contracts")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Target")
    table.add_column("Deadline")
    table.add_column("Days Remaining", justify="right")
    table.add_column("Reason")
    table.add_column("Location")

    for record in valid_until_records:
        days_remaining = record.days_remaining
        style = "yellow" if days_remaining <= 30 else None
        table.add_row(
            record.target,
            record.deadline.isoformat(),
            str(days_remaining),
            record.reason,
            f"{record.path.name}:{record.line}",
            style=style,
        )

    if not valid_until_records:
        table.add_row("(none)", "-", "-", "-", "-", style="dim")

    console.print(table)

    console.rule("Philosophy Violations")
    pv_table = Table(show_header=True, header_style="bold magenta")
    pv_table.add_column("Principle")
    pv_table.add_column("Count", justify="right")
    pv_table.add_column("Examples")

    for summary in sorted(
        principle_summaries.values(), key=lambda item: item.count, reverse=True
    ):
        pv_table.add_row(
            summary.principle,
            str(summary.count),
            "\n".join(summary.examples),
        )

    if not principle_summaries:
        pv_table.add_row("(none)", "0", "-", style="dim")

    console.print(pv_table)

    console.rule("Orphan Brand Colors")
    orphan_table = Table(show_header=True, header_style="bold magenta")
    orphan_table.add_column("Token")
    orphan_table.add_column("Hex")
    orphan_table.add_column("Used In")

    for usage in color_report.tokens:
        style = "red" if usage in color_report.orphans else None
        orphan_table.add_row(
            usage.token,
            usage.hex_value,
            ", ".join(usage.used_in) if usage.used_in else "(unused)",
            style=style,
        )

    if not color_report.tokens:
        orphan_table.add_row("(none)", "-", "-", style="dim")

    console.print(orphan_table)


def _render_plain_dashboard(
    valid_until_records: Sequence[ValidUntilRecord],
    principle_summaries: Dict[str, PrincipleSummary],
    color_report: ColorTokenReport,
) -> None:
    print("=== Temporal Contracts ===")
    if valid_until_records:
        for record in valid_until_records:
            marker = "!" if record.days_remaining <= 30 else "-"
            print(
                f"{marker} {record.target} – {record.deadline.isoformat()} "
                f"({record.days_remaining} days) : {record.reason} "
                f"[{record.path.name}:{record.line}]"
            )
    else:
        print("No valid_until decorators discovered.")

    print("\n=== Philosophy Violations ===")
    if principle_summaries:
        for summary in sorted(
            principle_summaries.values(), key=lambda item: item.count, reverse=True
        ):
            print(f"- {summary.principle}: {summary.count}")
            for example in summary.examples:
                print(f"    • {example}")
    else:
        print("No violations detected.")

    print("\n=== Orphan Brand Colors ===")
    if color_report.tokens:
        for usage in color_report.tokens:
            status = "unused" if usage in color_report.orphans else "used"
            print(
                f"- {usage.token} ({usage.hex_value}): "
                f"{', '.join(usage.used_in) if usage.used_in else status}"
            )
    else:
        print("No brand colors found in token file.")


def export_json(
    destination: Path,
    valid_until_records: Sequence[ValidUntilRecord],
    principle_summaries: Dict[str, PrincipleSummary],
    color_report: ColorTokenReport,
) -> None:
    """Export dashboard data to a JSON file at the specified destination."""
    payload = {
        "valid_until": [
            {
                "target": record.target,
                "deadline": record.deadline.isoformat(),
                "reason": record.reason,
                "path": str(record.path),
                "line": record.line,
                "days_remaining": record.days_remaining,
            }
            for record in valid_until_records
        ],
        "philosophy_violations": {
            principle: {
                "count": summary.count,
                "examples": summary.examples,
            }
            for principle, summary in principle_summaries.items()
        },
        "color_tokens": [
            {
                "token": usage.token,
                "hex": usage.hex_value,
                "used_in": usage.used_in,
                "unused": usage in color_report.orphans,
            }
            for usage in color_report.tokens
        ],
    }
    destination.write_text(json.dumps(payload, indent=2))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the dashboard CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to locate tests and sources (defaults to CWD)",
    )
    parser.add_argument(
        "--tests",
        type=Path,
        default=None,
        help="Override the tests directory (defaults to <root>/tests)",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Override the brand tokens file (defaults to repository deliverable)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to export the dashboard data as JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for decision decay dashboard CLI."""
    args = parse_args(argv)
    root = args.root.resolve()
    tests_root = (args.tests or (root / "tests")).resolve()

    if args.tokens is not None:
        tokens_path = args.tokens.resolve()
    else:
        tokens_path = root / "09_Client_Deliverables" / "Lantern_Logo_Implementation_Kit" / "lantern_tokens.json"

    valid_until_records = collect_valid_until_records(tests_root)
    principle_summaries = collect_philosophy_violations([root])
    color_report = collect_color_token_report(tokens_path)

    if args.json:
        export_json(args.json, valid_until_records, principle_summaries, color_report)

    render_dashboard(valid_until_records, principle_summaries, color_report)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
