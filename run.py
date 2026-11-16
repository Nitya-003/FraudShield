from __future__ import annotations

import argparse
import subprocess
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

os.environ.setdefault("FRAUDSHIELD_RATE_LIMIT", "600")

ROOT = Path(__file__).resolve().parent
FRONTEND_WEB_DIST = ROOT / "frontend-web" / "dist"


def main() -> None:
    parser = argparse.ArgumentParser(description="FraudShield Application Launcher")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run FastAPI server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind FastAPI server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Force rebuild of frontend-web before starting",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser on startup",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  FraudShield Multi-Channel Machine Learning Fraud Detection Platform")
    print("  Decision Engine & Modern React Frontend")
    print("=" * 68)

    # Check if frontend-web production build exists or rebuild requested
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    if args.build or not (FRONTEND_WEB_DIST / "index.html").exists():
        print("Note: Building frontend-web production bundle...")
        subprocess.run([npm_cmd, "run", "build", "--prefix", "frontend-web"], cwd=str(ROOT), check=True)

    target_url = f"http://localhost:{args.port}"
    print(f"Serving API and React Frontend at {target_url} ...")
    print(f"OpenAPI Documentation: {target_url}/docs")
    print("Press Ctrl+C to terminate.\n")

    if not args.no_browser:
        def _launch_browser() -> None:
            time.sleep(1.2)
            try:
                webbrowser.open(target_url)
            except Exception:
                pass

        threading.Thread(target=_launch_browser, daemon=True).start()

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.app:app",
        f"--host={args.host}",
        f"--port={args.port}",
    ]
    try:
        subprocess.run(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nFraudShield server shut down successfully.")


if __name__ == "__main__":
    main()
