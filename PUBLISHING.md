# Publishing HOLOS

Everything below needs **your** accounts/tokens — these are the steps that only you can run.

## 1. Push to GitHub

```bash
# Create an empty repo at github.com (public or private), then:
git remote add origin https://github.com/<YOUR_USER>/holos.git
git push -u origin main
```

Before pushing, replace the `USER` placeholders:
- `holos_core/Cargo.toml` → `repository = "https://github.com/<YOUR_USER>/holos"`
- `holos_py/README.md` and `README.md` → repository links
- `LICENSE` → confirm the copyright name

Once pushed, GitHub Actions (`.github/workflows/ci.yml`) automatically runs fmt/clippy/tests and
builds Python wheels for Linux/macOS/Windows.

## 2. Publish the Rust crate to crates.io

```bash
cargo login <YOUR_CRATES_IO_TOKEN>          # token from https://crates.io/settings/tokens
cargo publish --manifest-path holos_core/Cargo.toml
```

(`holos_core` has zero dependencies, so this is clean. Pick a final crate name first — check that
`holos_core` / `holos` is free on crates.io; rename in `Cargo.toml` if taken.)

## 3. Publish the Python package to PyPI

The easiest and most reliable way is **via CI on a version tag** (builds wheels for all platforms):

```bash
# Add your PyPI token as a repo secret named PYPI_API_TOKEN (Settings → Secrets → Actions),
# then tag a release:
git tag v0.1.0
git push origin v0.1.0
```

The `publish` job in the CI workflow uploads all wheels to PyPI automatically.

To publish manually from your machine instead:
```bash
pip install maturin
maturin publish --manifest-path holos_py/Cargo.toml   # prompts for your PyPI token
```

## Notes

- The package name `holos` must be free on PyPI — check https://pypi.org/project/holos/ and rename
  in `holos_py/pyproject.toml` + the `[lib] name` in `holos_py/Cargo.toml` if needed.
- Bump `version` in `holos_core/Cargo.toml`, `holos_py/Cargo.toml` and `holos_py/pyproject.toml`
  together for each release.
