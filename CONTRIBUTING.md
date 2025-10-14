# Contributing to Cloud Intelligence Platform

We love your input! We want to make contributing to the Cloud Intelligence Platform as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Git

### Local Development

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/cloud-intelligence-platform.git
   cd cloud-intelligence-platform
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run tests**
   ```bash
   pytest
   ```

6. **Start the development server**
   ```bash
   uvicorn api:app --reload
   ```

### Development Dependencies

Create a `requirements-dev.txt` file with development tools:

```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
isort>=5.12.0
mypy>=1.0.0
pre-commit>=3.0.0
```

## Code Style

We use several tools to maintain code quality:

### Python Code Formatting

- **Black**: Code formatter
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run formatting:
```bash
black .
isort .
flake8 .
mypy .
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality. Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new features
- Maintain or improve test coverage
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

Example test:
```python
def test_workload_scheduling():
    # Arrange
    workload = {"id": 1, "cpu_required": 2, "memory_required_gb": 4}
    
    # Act
    result = schedule_workload(workload)
    
    # Assert
    assert result.success is True
    assert result.vm_id is not None
```

## Documentation

### API Documentation

- Update OpenAPI specs when changing APIs
- Include examples in docstrings
- Update the API documentation in `docs/api/`

### User Documentation

- Update user guides when adding features
- Include screenshots for UI changes
- Update the README for significant changes

### Code Documentation

- Use clear, descriptive docstrings
- Follow Google-style docstrings
- Document complex algorithms and business logic

Example docstring:
```python
def schedule_workload(workload: Workload, scheduler_type: str) -> SchedulingResult:
    """Schedule a workload using the specified scheduler.
    
    Args:
        workload: The workload to schedule
        scheduler_type: Type of scheduler to use ('random', 'intelligent', etc.)
        
    Returns:
        SchedulingResult containing success status and assigned VM
        
    Raises:
        ValueError: If scheduler_type is not supported
        ResourceError: If no suitable VM is available
    """
```

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, etc.
- **Logs**: Relevant error messages or logs

### Feature Requests

Use the feature request template and include:

- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Alternative solutions considered
- **Additional Context**: Screenshots, mockups, etc.

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add intelligent workload scheduler
fix: resolve memory leak in cost calculator
docs: update API documentation for new endpoints
test: add integration tests for scheduler
refactor: simplify VM selection logic
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Branch Naming

Use descriptive branch names:

```
feature/intelligent-scheduler
fix/memory-leak-cost-calculator
docs/api-documentation-update
test/scheduler-integration-tests
```

## Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. Update version numbers
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release notes
6. Tag the release
7. Deploy to staging
8. Deploy to production

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Enforcement

Project maintainers are responsible for clarifying standards and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

## Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers at dev@cloud-intelligence.com

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Annual contributor highlights

Thank you for contributing to the Cloud Intelligence Platform! 🚀