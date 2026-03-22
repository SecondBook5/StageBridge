#!/usr/bin/env python3
"""Check SLURM job status for Snakemake."""

import subprocess
import sys

jobid = sys.argv[1]

try:
    result = subprocess.run(
        ["sacct", "-j", jobid, "--format=State", "--noheader", "--parsable2"],
        capture_output=True,
        text=True,
        timeout=30
    )

    status = result.stdout.strip().split("\n")[0].split("|")[0]

    if status in ["PENDING", "RUNNING", "REQUEUED", "RESIZING"]:
        print("running")
    elif status in ["COMPLETED"]:
        print("success")
    elif status in ["FAILED", "CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY"]:
        print("failed")
    else:
        print("running")  # Unknown status, assume still running

except Exception:
    print("running")  # On error, assume still running
