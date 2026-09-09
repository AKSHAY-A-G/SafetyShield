r"""Audit Milestone 0 without installing packages or loading models or credentials.

Run from the project root:
    .\venv\Scripts\python.exe -B scripts/check_environment.py

Exit codes: 0 = GPU environment ready, 1 = prerequisites failed,
2 = CPU checks passed but GPU readiness is blocked.
"""

import sys

# Also protect the existing venv when the caller forgets the -B flag.
sys.dont_write_bytecode = True

import argparse
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module, metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FOLDERS = (
    "venv", "config", "data/raw_videos", "data/frames", "data/dataset",
    "models", "src/camera", "src/detection", "src/tracking", "src/rules",
    "src/events", "src/database", "evidence", "tests",
)
REQUIRED_FILES = (
    ".env", ".gitignore", "README.md", "requirements.txt",
    "config/cameras.yaml", "config/rules.yaml", "config/zones.yaml",
)
PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "opencv-python": "cv2",
    "ultralytics": "ultralytics",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
}


def safe_error(error: Exception) -> str:
    """Report the error category; third-party messages can contain secrets."""
    return f"{type(error).__name__} (third-party message omitted for privacy)"


def inspect_package(distribution: str, module_name: str) -> tuple[dict, object]:
    """Check both installation metadata and a real import, without model loading."""
    result = {"version": None, "runtime_version": None, "ok": False, "error": None}
    try:
        result["version"] = metadata.version(distribution)
        module = import_module(module_name)
        result["runtime_version"] = getattr(module, "__version__", None)
        # Resolve the lazy YOLO class, but never construct/download a model.
        if distribution == "ultralytics":
            getattr(module, "YOLO")
        result["ok"] = True
        return result, module
    except Exception as error:
        result["error"] = safe_error(error)
        return result, None


def inspect_torch(torch_module: object) -> dict:
    """Check small CPU/CUDA calculations; GPU visibility alone is insufficient."""
    result = {
        "cpu_ok": False, "cuda_available": False, "cuda_build": None,
        "gpu_name": None, "gpu_vram_gib": None, "cuda_ok": False,
        "cpu_error": None, "cuda_error": None,
    }
    if torch_module is None:
        result["cpu_error"] = "PyTorch could not be imported."
        return result
    try:
        value = torch_module.ones(1, device="cpu")
        result["cpu_ok"] = (value + value).item() == 2.0
    except Exception as error:
        result["cpu_error"] = safe_error(error)
    try:
        result["cuda_build"] = torch_module.version.cuda
        result["cuda_available"] = torch_module.cuda.is_available()
        if result["cuda_available"]:
            properties = torch_module.cuda.get_device_properties(0)
            result["gpu_name"] = properties.name
            result["gpu_vram_gib"] = round(properties.total_memory / 1024**3, 2)
            value = torch_module.ones(1, device="cuda:0")
            result["cuda_ok"] = (value + value).item() == 2.0
            torch_module.cuda.synchronize()
        else:
            result["cuda_error"] = (
                "Installed PyTorch is a CPU-only build."
                if result["cuda_build"] is None
                else "PyTorch has CUDA support but cannot access a CUDA device."
            )
    except Exception as error:
        result["cuda_ok"] = False
        result["cuda_error"] = safe_error(error)
    return result


def inspect_driver() -> dict:
    """Read driver-reported hardware even when PyTorch has no CUDA support."""
    executable = shutil.which("nvidia-smi")
    result = {"gpus": [], "error": None}
    if executable is None:
        result["error"] = "nvidia-smi is not on PATH; driver information unavailable."
        return result
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        for line in completed.stdout.splitlines():
            name, memory, version = (part.strip() for part in line.split(",", 2))
            result["gpus"].append({
                "name": name, "vram_mib": float(memory), "driver_version": version,
            })
    except Exception as error:
        result["error"] = safe_error(error)
    return result


def assess_readiness(report: dict) -> tuple[int, str]:
    """Keep CPU usability separate from the planned GPU milestone gate."""
    if (
        not report["using_project_venv"] or report["missing_paths"]
        or not all(package["ok"] for package in report["packages"].values())
        or not report["torch"]["cpu_ok"]
    ):
        return 1, "NOT READY: fix the reported prerequisite failures."
    if not report["torch"]["cuda_ok"]:
        return 2, "CPU CHECKS PASSED; GPU NOT READY. Milestone 1 has not been started."
    return 0, "GPU ENVIRONMENT READY; model/video validation belongs to Milestone 1."


def collect_report() -> dict:
    report = {
        "python": platform.python_version(),
        "operating_system": platform.platform(),
        "executable": sys.executable,
        "using_project_venv": Path(sys.prefix).resolve() == (PROJECT_ROOT / "venv").resolve()
        and sys.prefix != sys.base_prefix,
        "missing_paths": [
            path + "/" for path in REQUIRED_FOLDERS
            if not (PROJECT_ROOT / path).is_dir()
        ] + [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()],
        "packages": {},
    }
    # Ultralytics/matplotlib may create settings on import. Confine those to
    # disposable temp storage, disable network checks and automatic installs.
    # Never load .env or print environment variables or third-party raw output.
    with tempfile.TemporaryDirectory(prefix="safetyshield-env-") as scratch:
        overrides = {
            "YOLO_CONFIG_DIR": scratch, "MPLCONFIGDIR": scratch,
            "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
        }
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                torch_module = None
                for distribution, module_name in PACKAGES.items():
                    package, module = inspect_package(distribution, module_name)
                    report["packages"][distribution] = package
                    if distribution == "torch":
                        torch_module = module
                report["torch"] = inspect_torch(torch_module)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    report["nvidia_driver"] = inspect_driver()
    report["exit_code"], report["readiness"] = assess_readiness(report)
    return report


def print_report(report: dict) -> None:
    print("SafetyShield - Milestone 0 environment check")
    print(f"Python: {report['python']}")
    print(f"Operating system: {report['operating_system']}")
    print(f"Interpreter: {report['executable']}")
    print(f"Using existing project venv: {report['using_project_venv']}")
    print(f"Project structure: {'PASS' if not report['missing_paths'] else 'FAIL'}")
    for path in report["missing_paths"]:
        print(f"  Missing: {path}")
    for name, package in report["packages"].items():
        version = package["runtime_version"] or package["version"] or "not installed"
        status = "PASS" if package["ok"] else "FAIL"
        print(f"{name}: {version} | import {status}")
        if package["error"]:
            print(f"  {package['error']}")
    compute = report["torch"]
    print(f"PyTorch CPU calculation: {'PASS' if compute['cpu_ok'] else 'FAIL'}")
    print(f"CUDA available to PyTorch: {compute['cuda_available']}")
    print(f"CUDA version used to build PyTorch: {compute['cuda_build'] or 'None'}")
    print(f"GPU name (PyTorch): {compute['gpu_name'] or 'unavailable'}")
    print(f"GPU VRAM (PyTorch, GiB): {compute['gpu_vram_gib']}")
    print(f"CUDA calculation: {'PASS' if compute['cuda_ok'] else 'NOT READY'}")
    for key in ("cpu_error", "cuda_error"):
        if compute[key]:
            print(f"  {compute[key]}")
    for gpu in report["nvidia_driver"]["gpus"]:
        print(f"NVIDIA driver: {gpu['name']}, {gpu['vram_mib']:g} MiB VRAM, "
              f"driver {gpu['driver_version']}")
    if report["nvidia_driver"]["error"]:
        print(f"NVIDIA driver query: {report['nvidia_driver']['error']}")
    print(f"Result: {report['readiness']}")
    print("No model, video, training, accuracy, FPS or full inference has been tested.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    args = parser.parse_args()
    try:
        report = collect_report()
    except Exception as error:
        failure = {"exit_code": 1, "error": safe_error(error)}
        print(json.dumps(failure) if args.json else f"Environment check failed: {failure['error']}")
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
