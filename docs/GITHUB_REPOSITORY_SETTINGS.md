# GitHub Repository Settings (Recommended)

Use this guide to configure repository quality gates that stay aligned with CI and contributor workflows.

## Branch Protection (`main`)

Enable the following settings:

- Require a pull request before merging
- Require approvals: 1 minimum
- Dismiss stale pull request approvals when new commits are pushed
- Require review from code owners
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require conversation resolution before merging
- Include administrators (recommended)

## Required Status Checks

Mark these checks as required. For matrix jobs, use the exact check names GitHub shows for each Python version.

- `lint` for Python 3.11 and 3.12
- `tests` for Python 3.11 and 3.12
- `type-check` for Python 3.11 and 3.12
- `packaging-sanity` for Python 3.11 and 3.12
- `docs-and-cli-sanity`

## Merge Strategy

Recommended:

- Allow squash merging
- Disallow merge commits
- Disallow rebase merging (optional, based on maintainer preference)
- Enable auto-delete head branches

## Security Settings

Enable these in repository `Settings > Security`:

- Dependency graph
- Dependabot alerts
- Dependabot security updates
- Secret scanning (and push protection, if available)
- Private vulnerability reporting

## Dependency Automation

Repository-level Dependabot updates are configured in `.github/dependabot.yml`.

- Keep Dependabot enabled for GitHub Actions and security update visibility.
- Review dependency PRs weekly and merge low-risk updates quickly.
- Prioritize security-labeled dependency PRs.

## Actions Settings

In `Settings > Actions`:

- Allow GitHub Actions and reusable workflows from verified creators
- Restrict workflow permissions to least privilege. CI uses `contents: read`; release uses `contents: write`, and PyPI publishing uses OIDC `id-token: write`.
- Require approval for first-time contributors (recommended)
