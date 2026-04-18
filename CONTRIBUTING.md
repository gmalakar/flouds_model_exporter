# Contributing

Thanks for your interest in contributing to flouds_model_exporter.

## Ground rules

- Be respectful and constructive in discussions and reviews.
- Keep pull requests focused and reasonably small.
- Include tests or clear validation steps for behavior changes.
- Update documentation when CLI behavior or defaults change.

## Development setup

1. Create and activate a virtual environment.
2. Install runtime dependencies:
   - `pip install -r requirements-prod.txt`
3. Install dev tooling extras:
   - `pip install -r requirements-dev.txt`
4. (Optional) Install package and CLI entrypoint:
   - `pip install -e .`
5. Enable pre-commit hooks:
   - `pre-commit install`

## Local checks

- Format and lint:
  - `pre-commit run --all-files`
- Run tests:
  - `pytest -q`

## Pull request checklist

- [ ] Code follows existing style and naming conventions.
- [ ] New and changed behavior is documented.
- [ ] Relevant tests pass locally.
- [ ] No secrets or tokens are included.
- [ ] Commit messages are clear and scoped.

## Reporting issues

Please include:

- OS and Python version
- Exact command used
- Full error output
- Steps to reproduce
- Expected vs actual behavior

Issue submissions should use the repository issue forms in `.github/ISSUE_TEMPLATE/`.

## Pull requests

Use `.github/PULL_REQUEST_TEMPLATE.md` and complete all checklist items.

## Maintainer notes

Repository protection and required-check guidance is documented in `docs/GITHUB_REPOSITORY_SETTINGS.md`.
Release and versioning workflow is documented in `docs/RELEASE_PROCESS.md`.

## Security issues

Do not open public issues for vulnerabilities. Follow the process in SECURITY.md.
