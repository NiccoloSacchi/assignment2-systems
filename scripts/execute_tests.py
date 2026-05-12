"""
Execute unit tests remotely.

Example usages:
  uv run modal run scripts/execute_tests.py
"""

from cs336_systems.modal_setup import app


@app.function(
    gpu="A10G",  # Tests for Triton kernels require a GPU
)
def run_pytest():
    import pytest
    import sys

    # Run pytest on the /root directory where your code is mounted
    # -s allows you to see print statements in the Modal logs
    exit_code = pytest.main(["-s", "/root/tests"])
    sys.exit(exit_code)


if __name__ == "__main__":
    with app.run():
        run_pytest.remote()
