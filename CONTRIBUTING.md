# Contributing to Plasmid Priority

Thank you for your interest in contributing to Plasmid Priority!

## Development Setup

### Prerequisites

- Python 3.12 or later
- [uv](https://github.com/astral-sh/uv) for dependency management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/plasmid-priority.git
   cd plasmid-priority
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync --dev
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

### Code Style

We use:
- **ruff** for linting and formatting
- **mypy** for type checking

Run the quality gate:
```bash
make quality
```

### Running Tests

Run the test suite:
```bash
make test
```

Run with coverage:
```bash
pytest tests/ --cov=src/plasmid_priority --cov-report=html
```

### Branch Pattern

When adding new prediction branches, follow the established pattern:

```
src/plasmid_priority/<branch_name>/
├── contracts.py    # Pydantic data contracts
├── specs.py        # Configuration specs
├── dataset.py     # Dataset preparation
├── features.py    # Feature engineering
├── train.py       # Model training
├── evaluate.py    # Model evaluation
├── calibration.py # Probability calibration
├── provenance.py  # Provenance tracking
├── report.py      # Reporting utilities
└── cli.py         # Command-line interface
```

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(geo_spread): add new geographic spread model
fix(model_audit): resolve calibration threshold error
docs(readme): update installation instructions
```

### Pull Request Process

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "feat(scope): description of changes"
   ```

3. Run the quality gate:
   ```bash
   make quality
   ```

4. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a Pull Request with:
   - Clear description of changes
   - Reference to related issues
   - Evidence of local testing

### Code Review Checklist

Before submitting a PR, ensure:
- [ ] Code passes `make quality`
- [ ] Tests pass `make test`
- [ ] New features have docstrings
- [ ] Type annotations are correct
- [ ] No hardcoded secrets or credentials
- [ ] Logging is used instead of print statements

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Full traceback if applicable

Use the issue templates when available.

## Questions?

Feel free to open an issue for questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
