Release process and PyPI publishing
=================================

1. Bump the version in `pyproject.toml` under `[project]`.
   - Follow SemVer: `MAJOR.MINOR.PATCH` (e.g., `0.1.1`).

2. Create a signed git tag and push it:

```bash
git commit -am "Bump version to 0.1.1"
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin main --follow-tags
```

3. CI will build artifacts and publish a GitHub Release when a `v*.*.*` tag is pushed.

4. To publish to PyPI from CI, add a `PYPI_API_TOKEN` secret in the repository settings:
   - Go to https://github.com/<owner>/flouds_model_exporter/settings/secrets/actions
   - Click New repository secret, name it `PYPI_API_TOKEN`, paste the token value.
   - In the release workflow, enable `publish_to_pypi` when dispatching the workflow.

5. Test uploads locally (optional):

```bash
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

6. After verifying on TestPyPI, publish to real PyPI via CI or:

```bash
python -m twine upload dist/* -u __token__ -p $PYPI_API_TOKEN
```
