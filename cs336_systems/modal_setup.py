from pathlib import Path
from typing import Optional

import modal

# Used to persist outputs, e.g. profile traces, of functions that run on Modal.
traces_volume = modal.Volume.from_name("cs336-systems-volume", create_if_missing=True)
TRACE_DIR = Path("/traces")

# Create an app will all needed packages.
app_image = (
    modal.Image.debian_slim()
    # Install cs336-basics before pip_install_from_pyproject tries (and fails)
    # to install it from PyPI.
    .apt_install("git")
    .run_commands(
      "git clone https://github.com/NiccoloSacchi/assignment1-basics /root/assignment1-basics",
      "pip install -e /root/assignment1-basics",
    )
    # This installs only the dependencies list, ignoring the rest, i.e. it does
    # not know that cs336-basics should be installed from a local path.
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source(
      # Imported by scripts/run_benchmark.py.
      "cs336_systems.benchmark",
      "cs336_systems.modal_setup",
    )
)
app = modal.App("cs336-systems", image=app_image)
