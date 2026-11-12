"""Run the internal API and public Streamlit UI as one supervised service."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(ROOT / ".env")
    # Keep the document API on loopback; the password-protected UI calls it.
    env = os.environ.copy()
    env["API_URL"] = "http://127.0.0.1:8000"
    children: list[subprocess.Popen] = []

    def stop(_signum=None, _frame=None):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=ROOT, env=env,
        )
        children.append(api)
        deadline = time.monotonic() + 90
        while True:
            if api.poll() is not None:
                raise RuntimeError(f"API exited during startup ({api.returncode})")
            try:
                with urlopen(env["API_URL"] + "/livez", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("API startup timed out")
            time.sleep(0.25)
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py",
             "--server.address=0.0.0.0", f"--server.port={env.get('PORT', '10000')}",
             "--server.headless=true", "--browser.gatherUsageStats=false",
             "--server.maxUploadSize=25"],
            cwd=ROOT, env=env,
        )
        children.append(ui)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        return next((child.returncode or 1 for child in children if child.poll() is not None), 1)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
