#!/usr/bin/env python3
"""Run Atlas evaluation with automatic crash recovery and monitoring.

This script runs the evaluation in a loop, automatically resuming if it crashes.
It monitors progress and logs to a file for tracking.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_evaluation_with_resume(
    conversations_path: Path,
    config_path: Path,
    output_dir: Path,
    sample: int,
    seed: int = 42,
    max_retries: int = 10,
    check_interval: int = 60,
) -> int:
    """Run evaluation with automatic resume on crash.
    
    Args:
        conversations_path: Path to conversations JSONL file
        config_path: Path to Atlas config YAML
        output_dir: Output directory for results
        sample: Number of conversations to sample
        seed: Random seed for sampling
        max_retries: Maximum number of retry attempts
        check_interval: Seconds between progress checks
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    log_file = output_dir / "evaluation.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open log file
    log_fd = open(log_file, "a")
    
    def log(message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        print(log_line.strip())
        log_fd.write(log_line)
        log_fd.flush()
    
    log("=" * 70)
    log("Starting Atlas 400 Conversation Evaluation")
    log("=" * 70)
    log(f"Conversations: {conversations_path}")
    log(f"Config: {config_path}")
    log(f"Output: {output_dir}")
    log(f"Sample: {sample} conversations")
    log(f"Seed: {seed}")
    log("")
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/run_atlas_evaluation.py",
        "--conversations", str(conversations_path),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--sample", str(sample),
        "--seed", str(seed),
    ]
    
    retry_count = 0
    last_progress_time = time.time()
    
    while retry_count < max_retries:
        try:
            log(f"Starting evaluation run (attempt {retry_count + 1}/{max_retries})...")
            
            # Run evaluation
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Monitor process
            for line in process.stdout:
                log(line.rstrip())
                last_progress_time = time.time()
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                log("")
                log("=" * 70)
                log("✅ Evaluation completed successfully!")
                log("=" * 70)
                log_fd.close()
                return 0
            else:
                log(f"")
                log(f"⚠️  Evaluation exited with code {return_code}")
                log(f"   The script supports automatic resume - checking if we can continue...")
                
                # Check if output file exists (indicates some progress was made)
                sessions_file = output_dir / "atlas" / "sessions.jsonl"
                if sessions_file.exists():
                    session_count = sum(1 for _ in open(sessions_file))
                    log(f"   Found {session_count} completed sessions - will resume from remaining conversations")
                else:
                    log(f"   No sessions file found - will restart from beginning")
                
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = min(30 * retry_count, 300)  # Exponential backoff, max 5 min
                    log(f"   Waiting {wait_time} seconds before retry {retry_count + 1}...")
                    time.sleep(wait_time)
                else:
                    log(f"")
                    log("=" * 70)
                    log(f"❌ Evaluation failed after {max_retries} attempts")
                    log("=" * 70)
                    log_fd.close()
                    return return_code
                    
        except KeyboardInterrupt:
            log("")
            log("⚠️  Evaluation interrupted by user")
            log("   Progress has been saved - you can resume by running the same command")
            log_fd.close()
            return 130
        except Exception as e:
            log(f"")
            log(f"❌ Error running evaluation: {e}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(30 * retry_count, 300)
                log(f"   Waiting {wait_time} seconds before retry {retry_count + 1}...")
                time.sleep(wait_time)
            else:
                log(f"")
                log("=" * 70)
                log(f"❌ Evaluation failed after {max_retries} attempts")
                log("=" * 70)
                log_fd.close()
                return 1
    
    log_fd.close()
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Atlas evaluation with automatic crash recovery"
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        required=True,
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Atlas config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--sample",
        type=int,
        required=True,
        help="Number of conversations to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Maximum retry attempts (default: 10)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between progress checks (default: 60)",
    )
    
    args = parser.parse_args()
    
    return run_evaluation_with_resume(
        conversations_path=args.conversations,
        config_path=args.config,
        output_dir=args.output_dir,
        sample=args.sample,
        seed=args.seed,
        max_retries=args.max_retries,
        check_interval=args.check_interval,
    )


if __name__ == "__main__":
    sys.exit(main())

