# SIMSOPT codebase guide for AI agents

## Big picture architecture
- Python package lives in [src/simsopt](src/simsopt) with major submodules in [src/simsopt/_core](src/simsopt/_core), [src/simsopt/geo](src/simsopt/geo), [src/simsopt/field](src/simsopt/field), [src/simsopt/mhd](src/simsopt/mhd), [src/simsopt/objectives](src/simsopt/objectives), [src/simsopt/solve](src/simsopt/solve), and [src/simsopt/util](src/simsopt/util).
- Performance-critical pieces are in C++ with pybind11 bindings in [src/simsoptpp](src/simsoptpp). The extension module is built via CMake in [CMakeLists.txt](CMakeLists.txt).
- Third‑party C++ deps are vendored as git submodules under [thirdparty](thirdparty); CMake will attempt submodule update during builds.

## Build and install
- Build is driven by scikit-build-core (see [pyproject.toml](pyproject.toml)); standard workflow is `pip install .` from the repo root (see [tests/README.md](tests/README.md)).
- CMake uses aggressive optimization flags and may auto‑download Boost if not found (see [CMakeLists.txt](CMakeLists.txt)); beware of platform‑specific flags (e.g., Apple silicon paths).

## Tests
- Test layout mirrors the Python package layout; data files live in [tests/test_files](tests/test_files) (see [tests/README.md](tests/README.md)).
- Primary test runner is unittest from the [tests](tests) directory; project scripts [run_tests](run_tests) and [run_tests_mpi](run_tests_mpi) show canonical invocations.

## Lint/format
- Ruff is the preferred linter (see [run_ruff](run_ruff) and ruff settings in [pyproject.toml](pyproject.toml)).
- Auto‑formatting uses autopep8 + flake8 via [run_autopep](run_autopep); note the project’s ignore list.

## Docs and examples
- Docs are built with Sphinx; canonical commands and directives are in [docs/README.md](docs/README.md).
- Example scripts are organized by difficulty in [examples](examples); overview in [examples/README.md](examples/README.md).

## Optional dependencies and integrations
- Feature‑gated extras (e.g., `SPEC`, `MPI`, `VIS`, `DOCS`, `ALGS`) are defined in [pyproject.toml](pyproject.toml). Ensure code paths that rely on these are guarded appropriately.

## Common change locations
- Python APIs and objective logic: [src/simsopt](src/simsopt).
- C++/pybind implementations: [src/simsoptpp](src/simsoptpp) and build wiring in [CMakeLists.txt](CMakeLists.txt).