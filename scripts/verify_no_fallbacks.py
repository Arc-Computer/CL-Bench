#!/usr/bin/env python
"""Verify all artifacts are free from fallbacks and placeholders."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

FALLBACK_PATTERNS = [
    r"FALLBACK-",
    r"fallback",
    r"placeholder",
    r"PLACEHOLDER",
    r"TODO",
    r"FIXME",
    r"XXX",
]

PLACEHOLDER_VALUES = [
    "placeholder",
    "PLACEHOLDER",
    "TODO",
    "FIXME",
    "xxx",
    "XXX",
]


def scan_jsonl_for_fallbacks(path: Path) -> List[Dict[str, Any]]:
    """Scan JSONL file for fallback patterns."""
    issues = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_str = json.dumps(record).lower()

                # Check for fallback patterns
                for pattern in FALLBACK_PATTERNS:
                    if re.search(pattern, record_str, re.IGNORECASE):
                        issues.append({
                            "file": str(path),
                            "line": line_num,
                            "type": "fallback_pattern",
                            "pattern": pattern,
                            "record_id": record.get("scenario_id") or record.get("conversation_id") or "unknown",
                        })

                # Check for placeholder values
                def check_value(value: Any, path: str = "") -> None:
                    if isinstance(value, str):
                        if value.lower() in PLACEHOLDER_VALUES:
                            issues.append({
                                "file": str(path),
                                "line": line_num,
                                "type": "placeholder_value",
                                "value": value,
                                "field_path": path,
                                "record_id": record.get("scenario_id") or record.get("conversation_id") or "unknown",
                            })
                    elif isinstance(value, dict):
                        for k, v in value.items():
                            check_value(v, f"{path}.{k}" if path else k)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            check_value(item, f"{path}[{i}]" if path else f"[{i}]")

                check_value(record)

            except json.JSONDecodeError:
                issues.append({
                    "file": str(path),
                    "line": line_num,
                    "type": "json_decode_error",
                    "error": "Invalid JSON",
                })

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory containing artifacts to scan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/verification_report.json"),
        help="Output path for verification report",
    )
    args = parser.parse_args()

    print(f"Scanning artifacts in {args.artifacts_dir}...")

    all_issues = []
    jsonl_files = list(args.artifacts_dir.rglob("*.jsonl"))

    for jsonl_file in jsonl_files:
        print(f"  Scanning {jsonl_file}...")
        issues = scan_jsonl_for_fallbacks(jsonl_file)
        all_issues.extend(issues)

    # Generate report
    report = {
        "total_files_scanned": len(jsonl_files),
        "total_issues": len(all_issues),
        "issues_by_type": {},
        "issues": all_issues,
    }

    for issue in all_issues:
        issue_type = issue["type"]
        report["issues_by_type"][issue_type] = report["issues_by_type"].get(issue_type, 0) + 1

    # Write report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    # Print summary
    print(f"\n✅ Verification complete!")
    print(f"  Files scanned: {report['total_files_scanned']}")
    print(f"  Issues found: {report['total_issues']}")
    for issue_type, count in report["issues_by_type"].items():
        print(f"    {issue_type}: {count}")

    if all_issues:
        print(f"\n⚠️  Issues found - see {args.output} for details")
        return 1
    else:
        print("\n✅ No fallbacks or placeholders found!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
