"""Meaningful failure-path tests; no ML dependencies or extra test runner needed."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT)
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def ready_report() -> dict:
    return {
        "using_project_venv": True,
        "missing_paths": [],
        "packages": {name: {"ok": True} for name in check.PACKAGES},
        "torch": {"cpu_ok": True, "cuda_ok": True},
    }


def fake_torch(cuda_available: bool = False) -> Mock:
    torch = Mock()
    torch.version.cuda = "test-cuda" if cuda_available else None
    torch.cuda.is_available.return_value = cuda_available
    torch.cuda.get_device_properties.return_value = SimpleNamespace(
        name="Test GPU", total_memory=4 * 1024**3,
    )
    value = Mock()
    value.__add__ = Mock(return_value=SimpleNamespace(item=lambda: 2.0))
    torch.ones.return_value = value
    return torch


class EnvironmentCheckTests(unittest.TestCase):
    def test_cpu_only_build_is_not_gpu_ready(self):
        result = check.inspect_torch(fake_torch())
        self.assertTrue(result["cpu_ok"])
        self.assertFalse(result["cuda_ok"])
        self.assertIn("CPU-only", result["cuda_error"])
        report = ready_report()
        report["torch"] = result
        self.assertEqual(check.assess_readiness(report)[0], 2)

    def test_cuda_device_must_execute_a_calculation(self):
        torch = fake_torch(True)
        torch.cuda.synchronize.side_effect = RuntimeError("unsupported CUDA kernel")
        result = check.inspect_torch(torch)
        self.assertTrue(result["cuda_available"])
        self.assertFalse(result["cuda_ok"])
        self.assertIn("RuntimeError", result["cuda_error"])

    def test_working_cuda_reports_gpu_and_vram(self):
        result = check.inspect_torch(fake_torch(True))
        self.assertTrue(result["cuda_ok"])
        self.assertEqual(result["gpu_name"], "Test GPU")
        self.assertEqual(result["gpu_vram_gib"], 4.0)
        self.assertEqual(check.assess_readiness(ready_report())[0], 0)

    def test_cuda_build_without_visible_gpu_has_distinct_diagnosis(self):
        torch = fake_torch()
        torch.version.cuda = "test-cuda"
        self.assertIn("cannot access", check.inspect_torch(torch)["cuda_error"])

    def test_wrong_interpreter_missing_structure_or_package_blocks_readiness(self):
        for cause in ("interpreter", "structure", "package", "cpu"):
            with self.subTest(cause=cause):
                report = ready_report()
                if cause == "interpreter":
                    report["using_project_venv"] = False
                elif cause == "structure":
                    report["missing_paths"] = ["config/cameras.yaml"]
                elif cause == "package":
                    report["packages"]["opencv-python"]["ok"] = False
                else:
                    report["torch"]["cpu_ok"] = False
                self.assertEqual(check.assess_readiness(report)[0], 1)

    def test_missing_torch_is_reported_without_crashing(self):
        self.assertFalse(check.inspect_torch(None)["cpu_ok"])

    def test_missing_package_is_reported(self):
        with patch.object(check.metadata, "version", side_effect=check.metadata.PackageNotFoundError):
            result, module = check.inspect_package("torch", "torch")
        self.assertFalse(result["ok"])
        self.assertIsNone(module)
        self.assertIn("PackageNotFoundError", result["error"])

    def test_import_error_does_not_leak_credentials(self):
        secret = "rtsp://test-user:fake-test-password@example.invalid/stream"
        with patch.object(check.metadata, "version", return_value="1.0"), patch.object(
            check, "import_module", side_effect=ImportError(secret)
        ):
            result, _ = check.inspect_package("opencv-python", "cv2")
        self.assertFalse(result["ok"])
        self.assertNotIn(secret, str(result))
        self.assertNotIn("fake-test-password", str(result))
        self.assertIn("ImportError", result["error"])

    def test_driver_query_reports_hardware_independently_of_torch(self):
        output = SimpleNamespace(stdout="NVIDIA GeForce GTX 1650, 4096, 610.47\n")
        with patch.object(check.shutil, "which", return_value="nvidia-smi"), patch.object(
            check.subprocess, "run", return_value=output
        ):
            result = check.inspect_driver()
        self.assertEqual(result["gpus"][0]["vram_mib"], 4096)
        self.assertIsNone(result["error"])

    def test_missing_driver_tool_is_nonfatal(self):
        with patch.object(check.shutil, "which", return_value=None):
            result = check.inspect_driver()
        self.assertEqual(result["gpus"], [])
        self.assertIn("not on PATH", result["error"])

    def test_driver_timeout_is_reported(self):
        with patch.object(check.shutil, "which", return_value="nvidia-smi"), patch.object(
            check.subprocess, "run", side_effect=check.subprocess.TimeoutExpired("nvidia-smi", 10)
        ):
            result = check.inspect_driver()
        self.assertIn("TimeoutExpired", result["error"])


if __name__ == "__main__":
    unittest.main()
