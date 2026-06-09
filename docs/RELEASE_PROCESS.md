# Release Process

Use this guide to run tag-driven releases, validate versioning, and publish to PyPI.

## Release Triggers

- Push a semantic version tag (`vX.Y.Z`) to trigger the release workflow and publish to PyPI.
- Or run the workflow manually from GitHub Actions (`workflow_dispatch`).

Workflow file: `.github/workflows/release.yml`

There is no separate PyPI publish workflow. GitHub Release creation and PyPI publishing both happen from `release.yml`.

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

6. Workflow builds `sdist` + `wheel`, validates metadata via `twine check`, uploads artifacts, creates a GitHub Release, and publishes the package to PyPI.

For tag-triggered releases, workflow validation enforces that tag `vX.Y.Z` exactly matches `project.version` in `pyproject.toml`.

## Manual Re-Release

If a release job needs to be re-run for an existing tag:

1. Open Actions -> Release -> Run workflow.
2. Set `tag` to the existing tag (for example `v0.1.1`).
3. Set `publish_to_pypi=true` only if the version has not already been published to PyPI.

Manual release runs also validate that the supplied tag matches `project.version` in `pyproject.toml`.

## PyPI Publish

Tag-triggered releases publish to PyPI automatically.

Manual workflow runs can also publish to PyPI:

- Set `publish_to_pypi=true` in the workflow input.
- Use this for an existing tag whose PyPI publish step did not run or failed before upload.
- Do not try to republish a version that already exists on PyPI; PyPI rejects duplicate version uploads.

Configure PyPI Trusted Publisher before tag pushes. If trusted publishing is not configured, the PyPI publish job will fail while GitHub Release creation can still succeed.

### PyPI Trusted Publisher Setup (Exact Fields)

In PyPI, open your project settings and add a new trusted publisher with:

- Owner: your GitHub org or username (for example `gmalakar`)
- Repository name: `flouds_model_exporter`
- Workflow filename: `release.yml`
- Environment name: `pypi`

The GitHub workflow uses `environment: pypi` on the PyPI publishing job. The PyPI trusted publisher environment must match that value exactly.

Recommended verification:

1. Save trusted publisher settings in PyPI.
2. Push a new release tag, or run `Release` manually with `publish_to_pypi=true` for an unpublished existing tag.
3. Confirm `publish-pypi` job succeeds and distribution appears in PyPI release history.

## Notes

- Keep tags immutable after release.
- Prefer patch releases for bug fixes and minor releases for new features.
- Ensure CLI examples in docs use canonical hyphenated flags.
- Keep release workflow actions on Node.js 24-compatible major versions.
