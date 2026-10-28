#!/usr/bin/env python3
"""One-command Google Colab setup and runner for RP-1.

Why this file is not called ``setup.py``
----------------------------------------
``setup.py`` is reserved by setuptools. This repository uses ``pyproject.toml``
and is installed with ``pip install -e . --no-deps``; adding a root ``setup.py``
would change packaging behaviour. This script is a Colab bootstrap, not a
package definition.

Typical use in Colab
--------------------
Two steps::

    !git clone https://github.com/Anurag71021/RP1-Leakproof-Material-Screening.git
    %cd RP1-Leakproof-Material-Screening
    !python colab_setup.py --run --app

Or one shell step::

    !git clone https://github.com/Anurag71021/RP1-Leakproof-Material-Screening.git \
      && cd RP1-Leakproof-Material-Screening \
      && python colab_setup.py --run --app

What ``--run`` does
-------------------
1. Creates a Python 3.10 virtual environment in ``.venv`` (Colab's default
   Python is too new for the pinned Matbench stack).
2. Installs the exact dependency set verified for this project.
3. Installs ``matbench`` with ``--no-deps`` to bypass its old matminer pin.
4. Regenerates the two gitignored inputs: the raw dataset cache and the
   deployment model / applicability-domain artefacts.
5. Runs the full test suite and the ``import app`` smoke test.

``--app`` additionally starts Streamlit and a temporary Cloudflare tunnel and
prints a public URL. ``--full`` rebuilds the entire scientific pipeline first.

Only the Python standard library is used here, so the script runs under Colab's
system Python before the project environment exists.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "bin" / "python"
REPORTS = ROOT / "reports"
ARTIFACT = ROOT / "artifacts" / "model_log_gvrh.joblib"

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

# Exact versions verified on macOS arm64 and available as Linux x86_64 wheels.
# Pinning matters: newer matminer/pymatgen releases can pull NumPy 2.x, which
# changes numerical behaviour and may break older compiled dependencies.
PINNED = [
    "numpy==1.26.4",
    "pandas==2.2.3",
    "scikit-learn==1.7.2",
    "scipy==1.15.3",
    "pyyaml>=6.0",
    "matplotlib>=3.6,<4.0",
    "seaborn>=0.12,<1.0",
    "streamlit==1.65.0",
    "pytest>=7.4",
    "joblib>=1.3",
    "matminer==0.9.3",
    "pymatgen==2025.10.7",
]

FULL_PIPELINE = [
    "build_dataset.py",
    "run_benchmark.py",
    "y_randomize.py",
    "run_diagnostics.py",
    "finalize_gate.py",
    "run_analysis.py",
    "build_literature.py",
    "train_surface_model.py",
    "build_screening.py",
]


def log(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def run(cmd: list[object], *, dry_run: bool = False, **kwargs: object) -> None:
    """Run a command from the repository root, printing it first."""
    parts = [str(part) for part in cmd]
    print("+ " + " ".join(parts), flush=True)
    if dry_run:
        return
    subprocess.run(parts, cwd=ROOT, check=True, **kwargs)


def ensure_uv(dry_run: bool) -> str:
    """Return a usable ``uv`` executable, installing it if necessary."""
    uv = shutil.which("uv")
    if uv:
        return uv
    if dry_run:
        return "uv"
    log("Installing uv")
    run([sys.executable, "-m", "pip", "install", "-q", "uv"], dry_run=False)
    uv = shutil.which("uv")
    if uv:
        return uv
    candidate = Path(sys.executable).parent / "uv"
    if candidate.exists():
        return str(candidate)
    raise SystemExit("Could not install or locate `uv`.")


def _venv_is_py310() -> bool:
    """Return True only if the existing .venv is Python 3.10."""
    if not VENV_PY.exists():
        return False
    try:
        result = subprocess.run(
            [str(VENV_PY), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return result.stdout.strip() == "3.10"


def ensure_environment(uv: str, args: argparse.Namespace) -> None:
    """Create the Python 3.10 environment and install the pinned stack."""
    log("Python 3.10 environment")
    run([uv, "python", "install", "3.10"], dry_run=args.dry_run)

    if VENV.exists() and not _venv_is_py310() and not args.dry_run:
        log("Existing .venv is not Python 3.10; recreating it")
        shutil.rmtree(VENV)

    if args.recreate_venv and VENV.exists() and not args.dry_run:
        log("Removing existing .venv")
        shutil.rmtree(VENV)

    if not VENV_PY.exists():
        run([uv, "venv", "--python", "3.10", ".venv"], dry_run=args.dry_run)
    else:
        print(f"Reusing existing environment: {VENV}")

    log("Installing pinned dependencies")
    run(
        [uv, "pip", "install", "--python", VENV_PY, *PINNED],
        dry_run=args.dry_run,
    )

    # Matbench 0.6 pins matminer 0.7.4, which has no modern Python build.
    log("Installing matbench without its broken dependency pins")
    run(
        [uv, "pip", "install", "--python", VENV_PY, "--no-deps", "matbench==0.6"],
        dry_run=args.dry_run,
    )

    log("Installing the local package")
    run(
        [uv, "pip", "install", "--python", VENV_PY, "-e", ".", "--no-deps"],
        dry_run=args.dry_run,
    )

    log("Verifying imports")
    run(
        [
            VENV_PY,
            "-c",
            "import matbench, matminer, pymatgen, sklearn, numpy; "
            "print('OK numpy', numpy.__version__)",
        ],
        dry_run=args.dry_run,
    )


def build_minimal(args: argparse.Namespace) -> None:
    """Regenerate the dataset cache and the deployment artefacts."""
    log("Dataset and deployment artefacts")
    run([VENV_PY, "scripts/build_dataset.py"], dry_run=args.dry_run)

    if args.force or args.dry_run or not ARTIFACT.exists():
        run([VENV_PY, "scripts/build_screening.py"], dry_run=args.dry_run)
    else:
        print(
            f"Deployment artefact already present: {ARTIFACT}\n"
            "Use --force to rebuild it."
        )


def build_full(args: argparse.Namespace) -> None:
    """Run the complete scientific pipeline in the documented order."""
    log("Full pipeline (this takes roughly 20-40 minutes on Colab CPU)")
    for script in FULL_PIPELINE:
        run([VENV_PY, f"scripts/{script}"], dry_run=args.dry_run)


def run_tests(args: argparse.Namespace) -> None:
    """Run the five required test suites and the import smoke test."""
    log("Tests")
    run([VENV_PY, "-m", "pytest", "-q"], dry_run=args.dry_run)
    run(
        [VENV_PY, "-c", "import app; print('APP_IMPORT_OK')"],
        dry_run=args.dry_run,
    )


def ensure_cloudflared(args: argparse.Namespace) -> Path:
    """Download the Cloudflare quick-tunnel binary if it is not present."""
    destination = ROOT / "cloudflared"
    if destination.exists():
        return destination
    if args.dry_run:
        log("Would download cloudflared")
        return destination
    if platform.system() != "Linux":
        raise SystemExit(
            "--app is intended for Colab/Linux. Run the app locally with "
            "`streamlit run app.py` instead."
        )
    log("Downloading cloudflared")
    urllib.request.urlretrieve(CLOUDFLARED_URL, destination)
    destination.chmod(0o755)
    return destination


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/_stcore/health", timeout=2
        ) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def launch_app(args: argparse.Namespace) -> None:
    """Start Streamlit and a Cloudflare tunnel, then print the public URL."""
    if not ARTIFACT.exists() and not args.dry_run:
        log("Deployment artefact missing; building it first")
        build_minimal(args)

    cloudflared = ensure_cloudflared(args)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stream_log_path = REPORTS / "colab_streamlit.log"
    tunnel_log_path = REPORTS / "colab_tunnel.log"

    if args.dry_run:
        log("Would start Streamlit and cloudflared")
        return

    log(f"Starting Streamlit on port {args.port}")
    stream_log = stream_log_path.open("w", encoding="utf-8")
    subprocess.Popen(
        [
            str(VENV / "bin" / "streamlit"),
            "run",
            "app.py",
            "--server.port",
            str(args.port),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
        stdout=stream_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if _healthy(args.port):
            break
        time.sleep(2)
    else:
        print(stream_log_path.read_text(encoding="utf-8", errors="ignore")[-3000:])
        raise SystemExit(
            f"Streamlit did not become healthy on port {args.port}. "
            f"See {stream_log_path}."
        )

    log("Starting Cloudflare tunnel")
    tunnel_log = tunnel_log_path.open("w", encoding="utf-8")
    subprocess.Popen(
        [
            str(cloudflared),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{args.port}",
            "--no-autoupdate",
        ],
        cwd=ROOT,
        stdout=tunnel_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + args.tunnel_timeout
    url = None
    while time.time() < deadline:
        text = tunnel_log_path.read_text(encoding="utf-8", errors="ignore")
        match = TUNNEL_RE.search(text)
        if match:
            url = match.group(0)
            break
        time.sleep(2)

    if url:
        print("\n" + "=" * 70)
        print(f"RP-1 app is live at: {url}")
        print("Keep this Colab tab and runtime alive while using the app.")
        print("=" * 70 + "\n")
    else:
        print(tunnel_log_path.read_text(encoding="utf-8", errors="ignore")[-3000:])
        raise SystemExit(
            f"Cloudflare did not return a URL within {args.tunnel_timeout}s. "
            f"See {tunnel_log_path}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up and run RP-1 in Google Colab.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python colab_setup.py --run --app\n"
            "  python colab_setup.py --run --skip-tests\n"
            "  python colab_setup.py --full --app\n"
            "  python colab_setup.py --dry-run --run --app\n"
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="build the dataset + deployment artefacts, then run tests",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="rebuild the entire scientific pipeline, then run tests",
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="launch Streamlit and print a Cloudflare tunnel URL",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="skip pytest and the app import check"
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild artefacts even if they exist"
    )
    parser.add_argument(
        "--recreate-venv", action="store_true", help="delete and recreate .venv"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print commands without executing them"
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Streamlit port (default: 8501)"
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=90,
        help="seconds to wait for Streamlit to become healthy (default: 90)",
    )
    parser.add_argument(
        "--tunnel-timeout",
        type=int,
        default=90,
        help="seconds to wait for a Cloudflare URL (default: 90)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log(f"RP-1 Colab setup in {ROOT}")

    if platform.system() == "Linux" and not Path("/content").exists():
        print(
            "Note: this does not look like Google Colab. The script still works "
            "on Linux, but --app expects Colab networking."
        )

    uv = ensure_uv(args.dry_run)
    ensure_environment(uv, args)

    if args.full:
        build_full(args)
    elif args.run:
        build_minimal(args)

    if (args.full or args.run) and not args.skip_tests:
        run_tests(args)

    if args.app:
        launch_app(args)
    elif not (args.full or args.run):
        print(
            "\nEnvironment is ready. Next:\n"
            "  python colab_setup.py --run --app\n"
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"\nCommand failed with exit code {exc.returncode}: {exc.cmd}"
        ) from exc
