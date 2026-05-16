"""
Execute unit tests remotely.

Example usages:
  uv run modal run scripts/execute_tests.py
  uv run modal run scripts/execute_tests.py --k test_flash_forward_pass_pytorch
"""

from cs336_systems.modal_setup import app
import pytest


@app.function(
    gpu="A100",  # Tests for Triton kernels require a GPU
)
def run_pytest(file, k):
    # Run pytest on the /root/tests directory where the modal app has put the
    # tests to be executed.
    target = f"/root/{file}" if file else "/root/tests"
    args = [target, "-v"]
    if k:
        args.extend(["-k", k])

    pytest.main(args)


@app.local_entrypoint()
def main(file: str = "", k: str = ""):
    run_pytest.remote(file, k)
