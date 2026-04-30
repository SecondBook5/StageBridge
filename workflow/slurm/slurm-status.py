#!/usr/bin/env python
"""Check SLURM job status for Snakemake."""

import subprocess
import sys


def get_status(jobid):
    """Get job status from SLURM."""
    try:
        result = subprocess.run(
            ["sacct", "-j", jobid, "--format=State", "--noheader", "--parsable2"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return "running"  # Assume running if can't check

        status = result.stdout.strip().split("\n")[0].split("|")[0]

        if status in ("COMPLETED",):
            return "success"
        elif status in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            return "running"
        else:
            return "failed"

    except Exception:
        return "running"


if __name__ == "__main__":
    jobid = sys.argv[1]
    print(get_status(jobid))
