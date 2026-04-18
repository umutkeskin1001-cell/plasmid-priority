#!/usr/bin/env python3
"""Bio transfer branch runner."""

from scripts.run_branch import main as _run_branch_main

if __name__ == "__main__":
    raise SystemExit(_run_branch_main(["--branch", "bio_transfer"]))
