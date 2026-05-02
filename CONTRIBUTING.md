# Contributing to StageBridge

Thank you for your interest in contributing to StageBridge!

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/SecondBook5/StageBridge.git
cd StageBridge
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check stagebridge/
```

## Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting
- Type hints are encouraged for public APIs
- Docstrings follow Google style

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Python version
- PyTorch version
- Steps to reproduce
- Expected vs actual behavior

## Questions

For questions about using StageBridge, please open a GitHub Discussion.
