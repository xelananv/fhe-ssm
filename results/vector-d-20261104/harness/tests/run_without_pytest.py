# SPDX-License-Identifier: MIT
"""Fallback runner for tests/test_harness.py when pytest is not installed (the project .venv is Python 3.9 and has no
pytest; the only copies in the repo are pytest 9.1.1, which requires Python >= 3.10). It builds the same fixtures as
conftest.py in a temp directory and calls every test_* function. Exit status 1 if any test fails.

  .venv/bin/python results/vector-d-20261104/harness/tests/run_without_pytest.py
"""
import inspect
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import conftest  # noqa: E402  (also puts the harness on sys.path)


class _Factory:
    def __init__(self, root):
        self.root = root

    def mktemp(self, name):
        import pathlib
        p = pathlib.Path(tempfile.mkdtemp(prefix=name, dir=self.root))
        return p


def main():
    import pathlib
    import test_harness as T
    root = tempfile.mkdtemp(prefix="rwkv7_harness_tests_")
    fac = _Factory(root)
    fixtures = {}
    unwrap = lambda f: getattr(f, "__wrapped__", f)
    fixtures["ref_and_ckpt"] = unwrap(conftest.ref_and_ckpt)(fac)
    fixtures["text_files"] = unwrap(conftest.text_files)(fac)
    failed = 0
    for name, fn in inspect.getmembers(T, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        kw = {}
        for p in inspect.signature(fn).parameters:
            kw[p] = pathlib.Path(tempfile.mkdtemp(dir=root)) if p == "tmp_path" else fixtures[p]
        t0 = time.time()
        try:
            fn(**kw)
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {name} ({time.time() - t0:.1f} s)", flush=True)
            traceback.print_exc()
    print(f"{failed} failed" if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
