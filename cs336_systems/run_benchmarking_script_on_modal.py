# Run with:
# uv run modal run /Users/niccolosacchi/assignment2-systems/cs336_systems/run_benchmarking_script_on_modal.py
import modal_setup

app = modal_setup.app()


@app.function(
    # gpu="T4",  # 16GB. OOO on large.
    # gpu="L4",  # 24GB. OOO on xl.
    gpu="A100",  # 80GB. No OOO, GPU memory peaked at ~38GB.
)
def run_benchmarking():
    import subprocess
    subprocess.run([
        "python", "benchmarking_script.py"
    ], check=True)


@app.local_entrypoint()
def main():
    run_benchmarking.remote()
