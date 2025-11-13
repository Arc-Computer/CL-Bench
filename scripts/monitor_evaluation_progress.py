#!/usr/bin/env python3
"""Real-time progress monitor for evaluation runs.

Tails JSONL output files and displays running progress, success rate, and ETA.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))


def count_results(file_path: Path) -> tuple[int, int]:
    """Count total and successful results in JSONL file."""
    total = 0
    successful = 0
    
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
                    try:
                        result = json.loads(line)
                        if result.get("overall_success", False):
                            successful += 1
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    
    return total, successful


def monitor_progress(file_path: Path, update_interval: int = 5) -> None:
    """Monitor progress by tailing JSONL file."""
    print(f"Monitoring: {file_path}")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    last_count = 0
    
    try:
        while True:
            total, successful = count_results(file_path)
            
            if total > last_count:
                success_rate = (successful / total * 100.0) if total > 0 else 0.0
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"Progress: {total} conversations | "
                    f"Success: {successful} ({success_rate:.1f}%) | "
                    f"Failed: {total - successful}"
                )
                last_count = total
            
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        total, successful = count_results(file_path)
        success_rate = (successful / total * 100.0) if total > 0 else 0.0
        print(f"Final: {total} conversations | Success: {successful} ({success_rate:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor evaluation progress")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file to monitor")
    parser.add_argument(
        "--update-interval",
        type=int,
        default=5,
        help="Update interval in seconds (default: 5)",
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return 1
    
    monitor_progress(args.input, args.update_interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())

