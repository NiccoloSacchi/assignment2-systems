import modal

def app():
  """Returns the Modal app for this assignment.
  
  The app image contains libraries that change very rarely (to avoid having to
  re-build).
  """
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
      .pip_install_from_pyproject("/Users/niccolosacchi/assignment2-systems/pyproject.toml")
      .add_local_python_source("benchmark")
  )
  return modal.App("cs336-systems", image=app_image)