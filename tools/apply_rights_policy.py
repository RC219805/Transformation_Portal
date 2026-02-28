#!/usr/bin/env python3
"""Apply deterministic rights/access policy flags to archive manifest entries."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from archive_governance_common import (  # pylint: disable=wrong-import-position
    atomic_write_text,
    deterministic_json_dumps,
    json_line,
)

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 2
EXIT_POLICY_ERROR = 3


class PolicyError(ValueError):
    """Raised when policy YAML is invalid."""


def _normalize_relpath(value: str) -> str:
    return value.replace("\\", "/")


def _iter_manifest_entries(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest JSONL line {line_number} must be an object")
            yield payload


def _normalize_flags(raw_flags: Any, *, context: str) -> list[str]:
    if not isinstance(raw_flags, list) or not raw_flags:
        raise PolicyError(f"{context} must be a non-empty list")
    normalized = sorted(set(str(flag).strip() for flag in raw_flags if str(flag).strip()))
    if not normalized:
        raise PolicyError(f"{context} must include at least one non-empty flag")
    return normalized


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PolicyError(f"Unable to parse policy file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PolicyError("Policy root must be a mapping")

    version = payload.get("version")
    if not isinstance(version, int) or version < 1:
        raise PolicyError("policy.version must be an integer >= 1")

    default_flags = _normalize_flags(payload.get("default_flags"), context="policy.default_flags")
    default_owner = str(payload.get("default_owner") or "UNSPECIFIED").strip() or "UNSPECIFIED"

    rules_raw = payload.get("rules")
    if rules_raw is None:
        rules_raw = []
    if not isinstance(rules_raw, list):
        raise PolicyError("policy.rules must be a list")

    normalized_rules: list[dict[str, Any]] = []
    for index, rule in enumerate(rules_raw):
        context = f"policy.rules[{index}]"
        if not isinstance(rule, dict):
            raise PolicyError(f"{context} must be an object")

        rule_id = str(rule.get("id") or "").strip()
        if not rule_id:
            raise PolicyError(f"{context}.id must be a non-empty string")

        flags = _normalize_flags(rule.get("flags"), context=f"{context}.flags")
        owner = str(rule.get("owner") or "").strip() or None

        path_glob = rule.get("path_glob")
        extension_in = rule.get("extension_in")
        relpath_regex = rule.get("relpath_regex")

        if path_glob is None and extension_in is None and relpath_regex is None:
            raise PolicyError(f"{context} must define at least one matcher")

        normalized_rule: dict[str, Any] = {
            "id": rule_id,
            "flags": flags,
            "owner": owner,
        }

        if path_glob is not None:
            if not isinstance(path_glob, str) or not path_glob.strip():
                raise PolicyError(f"{context}.path_glob must be a non-empty string")
            normalized_rule["path_glob"] = path_glob.strip()

        if extension_in is not None:
            if not isinstance(extension_in, list) or not extension_in:
                raise PolicyError(f"{context}.extension_in must be a non-empty list")
            ext_values = sorted(
                {str(value).strip().lower() for value in extension_in if isinstance(value, str) and str(value).strip()}
            )
            if not ext_values:
                raise PolicyError(f"{context}.extension_in must include string values")
            normalized_rule["extension_in"] = ext_values

        if relpath_regex is not None:
            if not isinstance(relpath_regex, str) or not relpath_regex.strip():
                raise PolicyError(f"{context}.relpath_regex must be a non-empty string")
            try:
                compiled_relpath_regex = re.compile(relpath_regex)
            except re.error as exc:
                raise PolicyError(f"{context}.relpath_regex is invalid: {exc}") from exc
            normalized_rule["relpath_regex"] = relpath_regex
            normalized_rule["_relpath_regex_compiled"] = compiled_relpath_regex

        normalized_rules.append(normalized_rule)

    return {
        "version": version,
        "default_flags": default_flags,
        "default_owner": default_owner,
        "rules": normalized_rules,
    }


def _rule_matches(entry: dict[str, Any], rule: dict[str, Any]) -> bool:
    relpath = _normalize_relpath(str(entry.get("relpath") or ""))
    extension = str(entry.get("extension") or "").lower()

    if "path_glob" in rule and not fnmatch.fnmatchcase(relpath, _normalize_relpath(str(rule["path_glob"]))):
        return False

    if "extension_in" in rule and extension not in set(rule["extension_in"]):
        return False

    compiled_relpath_regex = rule.get("_relpath_regex_compiled")
    if compiled_relpath_regex is None and "relpath_regex" in rule:
        compiled_relpath_regex = re.compile(str(rule["relpath_regex"]))
    if compiled_relpath_regex is not None and compiled_relpath_regex.search(relpath) is None:
        return False

    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", required=True, help="Input archive_manifest_v2.jsonl path")
    parser.add_argument("--policy-yaml", required=True, help="Policy YAML path")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL with rights flags")
    parser.add_argument("--out-summary", required=True, help="Output summary JSON path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest_jsonl)
    out_jsonl_path = Path(args.out_jsonl)
    out_summary_path = Path(args.out_summary)
    tmp_out_path = out_jsonl_path.with_name(f".{out_jsonl_path.name}.{uuid4().hex}.tmp")

    try:
        policy = _load_policy(Path(args.policy_yaml))
    except PolicyError as exc:
        print(f"Policy error: {exc}", file=sys.stderr)
        return EXIT_POLICY_ERROR

    rule_hits: Counter[str] = Counter()
    processed_count = 0

    try:
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_out_path.open("w", encoding="utf-8", newline="\n") as output_handle:
            for entry in _iter_manifest_entries(manifest_path):
                chosen_rule = "default"
                rights_flags = list(policy["default_flags"])
                owner = str(entry.get("owner") or policy["default_owner"])

                for rule in policy["rules"]:
                    if _rule_matches(entry, rule):
                        chosen_rule = str(rule["id"])
                        rights_flags = list(rule["flags"])
                        if rule.get("owner"):
                            owner = str(rule["owner"])
                        break

                updated = dict(entry)
                updated["rights_flags"] = sorted(set(rights_flags))
                updated["owner"] = owner
                output_handle.write(json_line(updated))

                rule_hits[chosen_rule] += 1
                processed_count += 1
        tmp_out_path.replace(out_jsonl_path)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except OSError as exc:
        print(f"Output error: {exc}", file=sys.stderr)
        return EXIT_POLICY_ERROR
    finally:
        if tmp_out_path.exists():
            tmp_out_path.unlink()

    summary_payload = {
        "schema_version": "tp.archive.rights.summary.v1",
        "policy_version": int(policy["version"]),
        "entry_count": processed_count,
        "rule_hit_counts": {key: int(value) for key, value in sorted(rule_hits.items())},
        "default_classification_count": int(rule_hits.get("default", 0)),
    }

    try:
        atomic_write_text(out_summary_path, deterministic_json_dumps(summary_payload, pretty=True) + "\n")
    except OSError as exc:
        print(f"Output error: {exc}", file=sys.stderr)
        return EXIT_POLICY_ERROR

    print(f"Applied rights policy to {processed_count} entries")
    print(f"Wrote {args.out_jsonl} and {args.out_summary}")
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
