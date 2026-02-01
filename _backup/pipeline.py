import subprocess
import sys

STEPS = [
    ("clean", ["python", "scripts/clean.py"]),
    ("activity", ["python", "scripts/activity.py"]),
    ("partition", ["python", "scripts/partition.py"]),
    ("train", ["python", "scripts/train.py"]),
    ("score", ["python", "scripts/score.py"]),
]

def run_step(name, cmd):
    print(f"\n=== {name.upper()} ===")
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    for name, cmd in STEPS:
        run_step(name, cmd)
    print("\n✅ Pipeline finished.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed (step exited with code {e.returncode}).")
        sys.exit(e.returncode)
