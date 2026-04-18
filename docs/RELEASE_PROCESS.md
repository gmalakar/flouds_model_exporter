# Release Process

Use this guide to run tag-driven releases, validate versioning, and optionally publish to PyPI.

## Release Triggers

- Push a semantic version tag (`vX.Y.Z`) to trigger the release workflow.
- Or run the workflow manually from GitHub Actions (`workflow_dispatch`).

Workflow file: `.github/workflows/release.yml`

## Standard Release Flow

1. Ensure `main` is green (CI checks passing).
2. Update `CHANGELOG.md` under `Unreleased`.
3. Bump `project.version` in `pyproject.toml`.
4. Commit the version/changelog update.
5. Create and push an annotated tag:

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

6. Workflow builds `sdist` + `wheel`, validates metadata via `twine check`, uploads artifacts, and creates a GitHub Release.

For tag-triggered releases, workflow validation enforces that tag `vX.Y.Z` exactly matches `project.version` in `pyproject.toml`.

## Manual Re-Release

If a release job needs to be re-run for an existing tag:

1. Open Actions -> Release -> Run workflow.
2. Set `tag` to the existing tag (for example `v0.1.1`).

## Optional PyPI Publish

Manual workflow runs support optional publish to PyPI:

- Set `publish_to_pypi=true` in the workflow input.
- Configure PyPI Trusted Publisher for this GitHub repository/workflow.

If trusted publishing is not configured, the PyPI publish job will fail while GitHub Release creation still succeeds.

### PyPI Trusted Publisher Setup (Exact Fields)

In PyPI, open your project settings and add a new trusted publisher with:

- Owner: your GitHub org or username (for example `goutam-malakar`)
- Repository name: `flouds_model_exporter`
- Workflow name: `Release`
- Workflow file path: `.github/workflows/release.yml`
- Environment name: leave empty unless you intentionally use a GitHub Environment gate

Recommended verification:

1. Save trusted publisher settings in PyPI.
2. Run `Release` workflow manually with `publish_to_pypi=true`.
3. Confirm `publish-pypi` job succeeds and distribution appears in PyPI release history.

## Notes

- Keep tags immutable after release.
- Prefer patch releases for bug fixes and minor releases for new features.
- Ensure CLI examples in docs use canonical hyphenated flags.
