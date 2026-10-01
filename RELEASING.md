# Releasing coxlm

1. Before the first public release, replace the `TODO` URLs under `[project.urls]` in `pyproject.toml`
   with the real repository.
2. Bump `__version__` in `src/coxlm/__init__.py` (the only place the version lives) and add a
   `CHANGELOG.md` entry.
3. Build and check:

   ```bash
   rm -rf dist
   uv build                       # dist/coxlm-X.Y.Z.tar.gz and dist/coxlm-X.Y.Z-py3-none-any.whl
   uvx twine check dist/*
   ```

4. Test the built wheel in a clean environment:

   ```bash
   uv venv /tmp/coxlm-check && VIRTUAL_ENV=/tmp/coxlm-check uv pip install dist/*.whl pytest
   /tmp/coxlm-check/bin/python -m pytest tests -q
   ```

5. Publish to TestPyPI first (token from https://test.pypi.org/manage/account/token/):

   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"
   uv pip install --index-url https://test.pypi.org/simple/ coxlm    # smoke-test the upload
   ```

6. Publish to PyPI:

   ```bash
   uv publish --token "$PYPI_TOKEN"
   ```

7. Tag the release: `git tag vX.Y.Z && git push --tags`.
