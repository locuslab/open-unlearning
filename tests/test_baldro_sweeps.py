"""Shell regression checks. Run with: python tests/test_baldro_sweeps.py"""

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT / "community/methods/BalDRO"
DRIVER = """
runs=0
python() {
    runs=$((runs + 1))
    printf '%s\\n' "$*"
    if [ "$runs" -eq "$SWEEP_FAIL_AT" ]; then
        return 7
    fi
    return 0
}
date() { printf '000000\\n'; }
export -f python date
export runs
exec bash "$@"
"""


class BalDROSweepTests(unittest.TestCase):
    def run_sweep(self, script, fail_at=0, wrapper=False):
        command = (
            [str(METHOD / "run.sh"), str(script.relative_to(METHOD / "scripts"))]
            if wrapper
            else [str(script)]
        )
        result = subprocess.run(
            ["bash", "-c", DRIVER, "sweep-test", *command],
            cwd=ROOT,
            env={**os.environ, "SWEEP_FAIL_AT": str(fail_at)},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.stderr, "")
        return result

    def test_sweeps_preserve_failures_and_finish_the_grid(self):
        scripts = sorted((METHOD / "scripts").glob("*/*.sh"))
        self.assertEqual(len(scripts), 12)
        for script in scripts:
            with self.subTest(script=script.name):
                success = self.run_sweep(script)
                self.assertEqual(success.returncode, 0)
                runs = success.stdout.splitlines()
                self.assertGreater(len(runs), 1)
                for fail_at in (1, len(runs)):
                    with self.subTest(fail_at=fail_at):
                        failure = self.run_sweep(script, fail_at=fail_at)
                        self.assertNotEqual(failure.returncode, 0)
                        self.assertEqual(failure.stdout.splitlines(), runs)

    def test_run_wrapper_propagates_sweep_status(self):
        for benchmark in ("tofu", "muse"):
            script = METHOD / "scripts" / benchmark / f"train_{benchmark}_drnpo.sh"
            for fail_at in (0, 1):
                with self.subTest(benchmark=benchmark, fail_at=fail_at):
                    result = self.run_sweep(script, fail_at=fail_at, wrapper=True)
                    self.assertEqual(result.returncode == 0, fail_at == 0)


if __name__ == "__main__":
    unittest.main()
