"""Execute project-overview-v3.ipynb in place, keeping Windows from idle sleep meanwhile.

Usage (repo root): .venv\\Scripts\\python.exe notebooks\\run_v3.py
Idle sleep suspends the workers while the clock keeps running, which corrupts every
`wall_seconds` the notebook records. Closing the lid still sleeps the machine.
"""
import ctypes
import subprocess
import sys
from pathlib import Path

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        rc = subprocess.call([
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--inplace", str(root / "notebooks" / "project-overview-v3.ipynb"),
            "--ExecutePreprocessor.kernel_name=gof-venv", "--ExecutePreprocessor.timeout=-1",
        ], cwd=root)
    finally:
        if sys.platform == "win32":
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    sys.exit(rc)
