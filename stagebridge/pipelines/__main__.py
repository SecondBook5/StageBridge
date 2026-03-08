"""Allow ``python -m stagebridge.pipelines`` to run the package CLI."""

from stagebridge.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
