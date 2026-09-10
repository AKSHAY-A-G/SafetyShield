# SafetyShield

Construction/factory computer vision safety monitoring prototype

Current stage: Milestone 1 automated checks complete; visual acceptance is
pending user review.

The existing Python 3.14.5 environment now uses torch 2.14.0+cu130 and
torchvision 0.29.0+cu130. A real CUDA matrix calculation passed on the GTX 1650,
the environment checker returned 0, all 11 tests passed and dependencies are
consistent. The previous CPU-only finding is preserved in the project plan.
Milestone 1 uses the official COCO-pretrained `yolo26n.pt` checkpoint to draw
person-only boxes and confidence labels on original-resolution recorded-video
frames. It does not perform tracking or PPE analysis.

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

Run Milestone 1 from the project root:

```powershell
.\venv\Scripts\python.exe -B scripts\run_person_detection.py --input data\raw_videos\cam_good_test.mp4
```

The input must be a readable local MP4. The command uses CUDA device 0,
`imgsz=640` and a configurable starting confidence threshold of 0.25. It writes
`outputs\cam_good_test_person_detected.mp4`, reports measured processing FPS and
PyTorch peak allocated GPU memory, then reopens the output to check its frame
size and readability. A missing input, invalid FPS, unavailable CUDA device or
unsupported output codec is reported as an error. Generated videos and model
weights are ignored by Git.

Do not recreate `venv`. Package changes require explicit authorization.
