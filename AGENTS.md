# SafetyShield engineering instructions

## Scope and milestone discipline

- Read `PROJECT_PLAN.md` before work. The supplied SafetyShield module document
  is the business reference; explicit user instructions govern execution scope.
- Work on one authorized milestone at a time. Do not implement unrelated or
  future modules just because the source document lists them.
- Current checkpoint: Milestone 1 complete; selected person-detection baseline
  is `yolo26n.pt`, `imgsz=960`, `conf=0.20`, subject to later labelled
  evaluation. These values remain configurable and are not scientific
  constants. PyTorch 2.14.0+cu130 and torchvision 0.29.0+cu130 execute CUDA on
  the GTX 1650; all 16 current unit tests and `pip check` pass. Recheck current
  results rather than assuming this observation is permanent.
- Do not start Milestone 2 until the user explicitly authorizes it. Stop this
  execution after closing Milestone 1.
- Every milestone needs a purpose, necessary files only, local verification,
  observable acceptance evidence and a Git commit before advancing. Report
  blocked checks; code running without errors is not evidence of CV quality.

## Existing environment and dependencies

- Reuse `venv` in the project root. Never create another/replacement environment
  or put project source, data, configuration or outputs inside `venv`.
- Do not modify the existing venv or install/update packages without explicit
  user authorization. Milestone 0 must not download models or datasets.
- Use `.\venv\Scripts\python.exe` explicitly for commands. Use `-B` for audit
  and test runs to avoid writing bytecode inside the existing environment.
- Keep requirements minimal. Add a dependency only when needed for the current
  milestone; do not freeze every installed transitive dependency into the file.
- Verify compatibility before any authorized Python/PyTorch/CUDA change.
  Driver visibility, PyTorch CUDA support and successful GPU execution are
  distinct checks. Do not assume a missing CUDA runtime requires a new venv.

## Hardware budget

- Design for Ryzen 5 4600H, 8 GB RAM and GTX 1650 with 4 GB VRAM on Windows.
- Prefer pretrained nano-class models. Begin with one feed and bounded queues,
  crop workloads and evidence buffers. Avoid simultaneously loading many
  models, large batches, large images or every camera.
- Initial PPE training candidates: `imgsz=640`, `batch=2`, `workers=2`,
  `cache=false`, `device=0` only after CUDA is verified. On CUDA OOM reduce
  batch to 1 first. Tune experimentally and report actual RAM/VRAM/FPS.
- Optional ROIs/tiling need configuration and a measured resource budget; do
  not run full-rate tiling across all cameras.

## Secrets and logging

- Never commit secrets. Store future RTSP URLs only in ignored `.env` and
  reference variable names in `config/cameras.yaml`.
- Never print credentials, full credential-bearing RTSP URLs, environment dumps
  or raw third-party exception text that could contain secrets. Redact logs
  and evidence metadata. Do not read or populate `.env` for environment audits.
- Disable package auto-install/download behaviour during diagnostic runs.
- Use clear logging for camera state, performance, detections and errors when
  those modules exist. Report failures explicitly instead of silently ignoring
  them; safe error categories may replace secret-bearing exception messages.
- Stage explicit project files, inspect staged changes and verify ignore rules
  before commits. Keep venv, credentials, raw videos, evidence, weights and
  databases out of commits. Never invent Git author identity.

## Vision and safety semantics

- Keep the original-resolution frame until worker crops have been extracted.
  Convert detector boxes to original coordinates and bound crops to the frame.
- Start with a pretrained general person/vehicle detector and a separate later
  helmet/vest detector. Do not train person detection from scratch initially.
- PPE states are PRESENT, MISSING and UNKNOWN. Small, occluded, blurred or
  low-confidence workers may be UNKNOWN. Non-detection does not prove MISSING.
  Define and validate explicit absence evidence before emitting missing PPE.
- Person-height/visibility/confidence thresholds are configurable and empirical.
  Do not present sample values as scientifically fixed or claim cropping
  recovers information absent from the original frame.
- Use per-track temporal evidence, minimum valid votes, expiry and event
  deduplication. UNKNOWN does not vote MISSING; one unreliable frame must not
  create a violation. Do not retain stale PPE states indefinitely.
- ByteTrack IDs are temporary within a camera/session. Never call them employee
  IDs or infer identity across cameras. Cross-camera Re-ID is future work.
- Configure polygons in `config/zones.yaml`; use worker bottom-centre points
  and one zone event per entry episode, with explicit re-entry behaviour.
- Moving cameras must not apply fixed polygons, static calibration, ground-plane
  distances or static-geometry speed rules.
- Image-space proximity is an experimental risk indicator. Never claim metres,
  physical speed or work height without calibrated geometry and validation.
- Never invent accuracy, training outcomes, saved lives or production capacity.
  Master-document metrics and claims are unverified until actually measured.

## Data, architecture and code quality

- Keep camera, detection, tracking, rules, events and database responsibilities
  separate using the existing folders. Avoid a giant script or unnecessary
  frameworks/abstraction. Use readable names, useful type hints and comments.
- Use configuration for paths and important thresholds. Keep model inference
  independent of event storage. SQLite comes before any cloud database;
  Streamlit comes after the vision pipeline works.
- Use transfer learning for PPE. Sample frames at configurable time intervals.
  Split by clip/time/camera before producing crops so related frames/crops do
  not leak across train/validation/test. Record dataset/checkpoint provenance.
- Evaluate helmet/vest precision and recall, UNKNOWN coverage and event errors
  by real person-size/visibility groups with sample counts. Keep test data
  held out. Never replace measured results with proposed targets.
- Preserve unrelated files and the original supplied document. Add only useful
  files/directories required for the current milestone.

## Verification and beginner explanations

- Run appropriate tests before claiming completion. For Milestone 0:

  ```powershell
  .\venv\Scripts\python.exe -B scripts/check_environment.py
  .\venv\Scripts\python.exe -B -m unittest discover -s tests -v
  .\venv\Scripts\python.exe -B -m pip check
  ```

- Checker exit codes: 0 = GPU environment ready, 1 = prerequisite failure,
  2 = CPU checks passed but GPU not ready. Unit-test mocks are not hardware,
  real-video or model validation. Record actual failed and skipped checks.
- For important new files, explain purpose, inputs, outputs, why needed, how
  to run/test them and common errors in beginner-friendly language.
- End a milestone with what changed, files, commands, expected output, actual
  test results, problems, learning points and the next milestone. Do not start
  the next milestone merely by announcing it.
