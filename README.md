# SafetyShield

Construction/factory computer vision safety monitoring prototype

Current stage: Milestone 0 complete - GPU readiness verified.

The existing Python 3.14.5 environment now uses torch 2.14.0+cu130 and
torchvision 0.29.0+cu130. A real CUDA matrix calculation passed on the GTX 1650,
the environment checker returned 0, all 11 tests passed and dependencies are
consistent. The previous CPU-only finding is preserved in the project plan.
Milestone 1 has not started.

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
The verified checker result is exit code 0: GPU environment checks passed.
Exit 2 means CPU checks passed but GPU readiness is blocked; exit 1 means
prerequisite/check failure.

Do not recreate `venv`. Package changes require explicit authorization.
