"""
Zero-dependency test runner fallback.

This repo's real test suite (tests/test_features.py, tests/test_model.py,
tests/test_drift.py) is written for pytest and is the intended way to run
tests (`python -m pytest tests/ -v`). This script exists only for
environments where `pytest` itself cannot be installed (e.g. no outbound
network access) — it discovers every `test_*` function in tests/*.py,
provides a minimal `pytest.fixture` / `pytest.mark.skipif` / `pytest.raises`
shim so the same test files run unmodified, and reports pass/fail counts.

Usage:
    python run_tests_no_pytest.py
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback
import types

sys.path.insert(0, os.path.dirname(__file__))


# --- minimal `pytest` shim -------------------------------------------------
class _SkipTest(Exception):
    pass


class _FixtureFunctionWrapper:
    def __init__(self, func, scope="function"):
        self.func = func
        self.scope = scope
        self._cache_key = None
        self._cache_value = None
        self._has_cache = False

    def resolve(self):
        if self.scope == "module" and self._has_cache:
            return self._cache_value
        value = self.func()
        if self.scope == "module":
            self._cache_value = value
            self._has_cache = True
        return value


def fixture(func=None, *, scope="function"):
    if func is not None:
        return _FixtureFunctionWrapper(func, scope=scope)

    def decorator(f):
        return _FixtureFunctionWrapper(f, scope=scope)

    return decorator


class _MarkDecorators:
    @staticmethod
    def skipif(condition, reason=""):
        def decorator(f):
            f._skip = condition
            f._skip_reason = reason
            return f

        return decorator

    class parametrize:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, f):
            return f


class _RaisesContext:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type} to be raised, but nothing was raised")
        if not issubclass(exc_type, self.exc_type):
            return False
        return True


def raises(exc_type):
    return _RaisesContext(exc_type)


pytest_shim = types.ModuleType("pytest")
pytest_shim.fixture = fixture
pytest_shim.mark = _MarkDecorators()
pytest_shim.raises = raises
pytest_shim.skip = lambda reason="": (_ for _ in ()).throw(_SkipTest(reason))
sys.modules["pytest"] = pytest_shim
# ---------------------------------------------------------------------------


def _resolve_fixtures(func, module, module_fixture_cache):
    import pathlib
    import shutil
    import tempfile

    sig = inspect.signature(func)
    kwargs = {}
    cleanup_dirs = []
    for name in sig.parameters:
        if name == "tmp_path":
            # Minimal stand-in for pytest's built-in tmp_path fixture.
            d = tempfile.mkdtemp(prefix="pytest_shim_")
            cleanup_dirs.append(d)
            kwargs[name] = pathlib.Path(d)
            continue

        candidate = getattr(module, name, None)
        if isinstance(candidate, _FixtureFunctionWrapper):
            if candidate.scope == "module":
                if name not in module_fixture_cache:
                    module_fixture_cache[name] = candidate.resolve()
                kwargs[name] = module_fixture_cache[name]
            else:
                kwargs[name] = candidate.resolve()
        elif candidate is None:
            raise RuntimeError(f"No fixture named '{name}' found for test {func.__name__}")
    return kwargs, cleanup_dirs


def run_module(path: str) -> tuple[int, int, list[str]]:
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    passed, failed = 0, 0
    failures = []
    module_fixture_cache: dict = {}

    test_funcs = [
        (name, obj) for name, obj in vars(module).items()
        if name.startswith("test_") and callable(obj) and not isinstance(obj, _FixtureFunctionWrapper)
    ]

    for name, func in test_funcs:
        skip = getattr(func, "_skip", False)
        if skip:
            print(f"  SKIP {name} ({getattr(func, '_skip_reason', '')})")
            continue
        cleanup_dirs = []
        try:
            kwargs, cleanup_dirs = _resolve_fixtures(func, module, module_fixture_cache)
            func(**kwargs)
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failures.append(f"{module_name}::{name}\n{traceback.format_exc()}")
            failed += 1
        finally:
            import shutil
            for d in cleanup_dirs:
                shutil.rmtree(d, ignore_errors=True)

    return passed, failed, failures


def main():
    tests_dir = os.path.join(os.path.dirname(__file__), "tests")
    test_files = sorted(
        os.path.join(tests_dir, f) for f in os.listdir(tests_dir)
        if f.startswith("test_") and f.endswith(".py")
    )

    total_passed, total_failed = 0, 0
    all_failures = []

    for path in test_files:
        print(f"\n=== {os.path.basename(path)} ===")
        p, f, failures = run_module(path)
        total_passed += p
        total_failed += f
        all_failures.extend(failures)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    if all_failures:
        print("\n--- Failure details ---")
        for f in all_failures:
            print(f)
    print("=" * 60)

    sys.exit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
