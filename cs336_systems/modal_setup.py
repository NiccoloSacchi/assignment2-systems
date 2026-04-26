from pathlib import Path

import modal

# Used to persist outputs, e.g. profile traces, of functions that run on Modal.
my_volume = modal.Volume.from_name("cs336-systems-volume", create_if_missing=True)
VOLUME_DIR = Path("/cs336_systems_volume")

# Create an app will all needed packages.
app_image = (
    modal.Image.debian_slim()
    # Install cs336-basics before pip_install_from_pyproject tries (and fails)
    # to install it from PyPI.
    .apt_install("git").run_commands(
        "git clone https://github.com/NiccoloSacchi/assignment1-basics /root/assignment1-basics",
        "pip install -e /root/assignment1-basics",
    )
    # -------------------------------------------------------------------
    # If you have assignment1-basics repo locally and are applying changes to
    # it, e.g. adding torch.profiler.record_function in the code, then you want
    # to copy the local code to the Modal image instead of cloning from GitHub.
    # This requires re-building the image so it is slower.
    # .add_local_dir(
    #   local_path="../assignment1-basics/cs336_basics",
    #   remote_path="/root/assignment1-basics/cs336_basics",
    #   copy=True,
    # )
    # .add_local_file(
    #   local_path="../assignment1-basics/pyproject.toml",
    #   remote_path="/root/assignment1-basics/pyproject.toml",
    #   copy=True,
    # )
    # .add_local_file(
    #   local_path="../assignment1-basics/README.md",
    #   remote_path="/root/assignment1-basics/README.md",
    #   copy=True,
    # )
    # .add_local_file(
    #   local_path="../assignment1-basics/uv.lock",
    #   remote_path="/root/assignment1-basics/uv.lock",
    #   copy=True,
    # )
    # .run_commands(
    #   "pip install -e /root/assignment1-basics",
    # )
    # -------------------------------------------------------------------
    # This installs only the dependencies list, ignoring the rest, i.e. it does
    # not know that cs336-basics should be installed from a local path.
    .pip_install_from_pyproject("pyproject.toml")
    # This does not require re-building the image, so it is quicker for
    # development.
    .add_local_python_source("cs336_systems")
)
app = modal.App("cs336-systems", image=app_image)
