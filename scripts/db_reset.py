#!/usr/bin/env python3
"""Restore the Postgres CRM database from a snapshot SQL file."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Sequence


DEFAULT_SNAPSHOT_PATH = pathlib.Path("artifacts/postgres_snapshot.sql")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset the CRM Postgres database from a snapshot.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help=f"Snapshot file to restore (default: {DEFAULT_SNAPSHOT_PATH})",
    )
    parser.add_argument(
        "--use-compose",
        choices=["auto", "always", "never"],
        default=os.getenv("DB_RESET_USE_COMPOSE", "auto"),
        help="Whether to run psql through docker compose exec.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Prints the executed command for debugging.",
    )
    return parser.parse_args(list(argv))


def resolve_db_credentials() -> Dict[str, str]:
    return {
        "host": os.getenv("DB_HOST", os.getenv("POSTGRES_HOST", "localhost")),
        "port": os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("DB_USER", os.getenv("POSTGRES_USER", "crm_app")),
        "password": os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "crm_password")),
        "dbname": os.getenv("DB_NAME", os.getenv("POSTGRES_DB", "crm_sandbox")),
    }


def should_use_compose(mode: str, host: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return host in {"localhost", "127.0.0.1", "db"}


def restore_via_compose(args: argparse.Namespace, creds: Dict[str, str], snapshot_path: pathlib.Path) -> int:
    command: List[str] = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        f"PGPASSWORD={creds['password']}",
        "db",
        "psql",
        "--username",
        creds["user"],
        "--dbname",
        creds["dbname"],
        "--single-transaction",
        "--echo-errors",
    ]
    if args.verbose:
        print("Running:", " ".join(command), file=sys.stderr)
    with snapshot_path.open("rb") as fh:
        result = subprocess.run(command, check=False, stdin=fh)
    return result.returncode


def restore_locally(args: argparse.Namespace, creds: Dict[str, str], snapshot_path: pathlib.Path) -> int:
    env = os.environ.copy()
    env["PGPASSWORD"] = creds["password"]
    command: List[str] = [
        "psql",
        "--host",
        creds["host"],
        "--port",
        str(creds["port"]),
        "--username",
        creds["user"],
        "--dbname",
        creds["dbname"],
        "--single-transaction",
        "--echo-errors",
    ]
    if args.verbose:
        print("Running:", " ".join(command), file=sys.stderr)
    with snapshot_path.open("rb") as fh:
        result = subprocess.run(command, check=False, env=env, stdin=fh)
    return result.returncode


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    snapshot_path: pathlib.Path = args.input
    if not snapshot_path.exists():
        print(f"Snapshot file not found: {snapshot_path}", file=sys.stderr)
        return 1

    creds = resolve_db_credentials()

    if should_use_compose(args.use_compose, creds["host"]):
        code = restore_via_compose(args, creds, snapshot_path)
    else:
        code = restore_locally(args, creds, snapshot_path)

    if code != 0:
        print("Database reset failed; see output above for details.", file=sys.stderr)
        return code

    print(f"Database restored from {snapshot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

