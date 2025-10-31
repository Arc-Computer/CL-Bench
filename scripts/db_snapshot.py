#!/usr/bin/env python3
"""Create a snapshot of the Postgres CRM database.

This utility uses either the locally installed `pg_dump` binary or
`docker compose exec db pg_dump` (when Docker is running the database)
to export the full schema + data into a plain SQL file.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Dict, List, Sequence


DEFAULT_SNAPSHOT_PATH = pathlib.Path("artifacts/postgres_snapshot.sql")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot the CRM Postgres database.")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help=f"Path to write the snapshot SQL (default: {DEFAULT_SNAPSHOT_PATH})",
    )
    parser.add_argument(
        "--use-compose",
        choices=["auto", "always", "never"],
        default=os.getenv("DB_SNAPSHOT_USE_COMPOSE", "auto"),
        help="Whether to run pg_dump through docker compose exec.",
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


def run_pg_dump_via_compose(args: argparse.Namespace, creds: Dict[str, str], output_path: pathlib.Path) -> int:
    command: List[str] = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        f"PGPASSWORD={creds['password']}",
        "db",
        "pg_dump",
        "--username",
        creds["user"],
        "--dbname",
        creds["dbname"],
        "--format=plain",
        "--no-owner",
        "--no-privileges",
        "--clean",
    ]
    if args.verbose:
        print("Running:", " ".join(command), file=sys.stderr)
    with output_path.open("wb") as fh:
        result = subprocess.run(command, check=False, stdout=fh)
    return result.returncode


def run_pg_dump_locally(args: argparse.Namespace, creds: Dict[str, str], output_path: pathlib.Path) -> int:
    env = os.environ.copy()
    env["PGPASSWORD"] = creds["password"]
    command: List[str] = [
        "pg_dump",
        "--host",
        creds["host"],
        "--port",
        str(creds["port"]),
        "--username",
        creds["user"],
        "--dbname",
        creds["dbname"],
        "--format=plain",
        "--no-owner",
        "--no-privileges",
        "--clean",
    ]
    if args.verbose:
        print("Running:", " ".join(command), file=sys.stderr)
    with output_path.open("wb") as fh:
        result = subprocess.run(command, check=False, env=env, stdout=fh)
    return result.returncode


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    creds = resolve_db_credentials()
    output_path: pathlib.Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if should_use_compose(args.use_compose, creds["host"]):
        code = run_pg_dump_via_compose(args, creds, output_path)
    else:
        code = run_pg_dump_locally(args, creds, output_path)

    if code != 0:
        print("Snapshot failed; see output above for details.", file=sys.stderr)
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
        return code

    print(f"Snapshot written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

