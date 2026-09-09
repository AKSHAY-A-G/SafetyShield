# SafetyShield

Construction/factory computer vision safety monitoring prototype

Current stage: Milestone 0 - project audit and environment check.

The existing Python environment passes package imports and a small CPU check.
GPU readiness is blocked: PyTorch and torchvision are CPU-only builds, although
the NVIDIA driver detects the GTX 1650. Milestone 1 has not started.

Read [PROJECT_PLAN.md](PROJECT_PLAN.md) for scope, source requirements,
milestones, measured audit results and troubleshooting. [AGENTS.md](AGENTS.md)
contains persistent engineering instructions.

Run these commands in PowerShell from the project root using the existing venv:

```powershell
.\venv\Scripts\python.exe -B scripts/check_environment.py
$LASTEXITCODE
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B -m pip check
```

The checker reports Python/OS, package imports, PyTorch CUDA support and driver
GPU details. It does not load credentials, install packages or load a model.
Exit code 2 currently means CPU checks passed but GPU readiness is blocked;
0 means GPU environment checks passed; 1 means prerequisite/check failure.

Do not recreate `venv`. Package changes require explicit authorization.
