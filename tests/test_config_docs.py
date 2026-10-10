import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_configuration_doc_is_up_to_date():
    r = subprocess.run([sys.executable, "scripts/gen-config-docs.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
