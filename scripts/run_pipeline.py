from pathlib import Path
import subprocess
import sys

# scripts/ is inside project root
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

PIPELINE_STEPS = [
    "clean.py",
    "activity.py",
    "partition_activity.py",
    "label.py",
    "train.py",
    "score.py",
]

def run_step(script_name: str):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print("\n" + "=" * 60)
    print(f"Running: python scripts/{script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline stopped. Step failed: {script_name}")


def main():
    print("Starting churn model pipeline")
    print("Project root:", PROJECT_ROOT)

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\nPipeline completed successfully ✔")


if __name__ == "__main__":
    main()
