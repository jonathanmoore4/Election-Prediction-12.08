"""Resume fitting, rebuild diagnostics, and audit with workspace-persistent caches.

Run with the repository's Python environment. NN_STUDY_CACHE and
NN_STUDY_WORKERS remain overridable. --postprocess-only skips fitting.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "all_elections"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("NN_STUDY_CACHE", str(ROOT / ".nn_study_cache"))
    OUT.mkdir(exist_ok=True)
    steps = [] if args.postprocess_only else [["run_all_elections_fast.py"]]
    steps += [["summarize_all_elections.py"], ["verify_results.py", "--directory", str(OUT)]]
    status = dict(cache=env["NN_STUDY_CACHE"], completed_steps=[])

    def save(state, step):
        status.update(state=state, step=step, updated_utc=datetime.now(timezone.utc).isoformat())
        (OUT / "resume_status.json").write_text(json.dumps(status, indent=2) + "\n")

    for step in steps:
        save("running", step[0])
        print(f"Running {step[0]}; cache: {env['NN_STUDY_CACHE']}", flush=True)
        try:
            subprocess.run([sys.executable, "-u", str(HERE / step[0]), *step[1:]],
                           env=env, check=True)
        except (subprocess.CalledProcessError, KeyboardInterrupt):
            save("interrupted_or_failed", step[0])
            raise
        status["completed_steps"].append(step[0])
    save("complete", None)


if __name__ == "__main__":
    main()
