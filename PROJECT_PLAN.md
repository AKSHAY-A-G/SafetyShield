# SafetyShield project plan

SafetyShield is a construction/factory computer vision safety monitoring
prototype. The first goal is a tested pipeline on one laptop and one video or
camera, followed by helmet/vest analysis, safety events, evidence and a simple
dashboard. Each milestone must demonstrate an observable result before the next
one begins.

Current stage: **Milestone 0 implemented and audited; GPU readiness blocked.**
The existing environment passes package imports and a CPU calculation, but its
PyTorch build cannot use CUDA. Milestone 1 has not started. No inference,
training, model accuracy or video processing performance has been measured.

## Source requirements and authority

Business reference: the supplied `SafetyShield V3 Modules.docx`, located at
`C:\Users\AKSHAY\Downloads\SafetyShield V3 Modules.docx` during this audit.
Its opening title is "NGXP SafetyShield AI - The Definitive Master Document
v2.0" and its closing summary says "NGXP SafetyShield v3". This version-label
inconsistency is preserved here instead of assuming a different specification.

The document contains 76 module entries across 10 categories and describes
construction sites with 80-120 cameras. The user's accompanying engineering
request sets the laptop prototype scope and milestone order. Module descriptions
are business requirements and future ideas, not instructions to execute every
feature in this pass.

The source's model accuracy, false-alert reduction, competitor superiority,
bandwidth, cloud pricing and "saved lives" claims are unverified source claims.
They are not SafetyShield test results or promises. Its closing summary refers
to PPE-014 through PPE-018 and HGT-014; PPE-015 through PPE-018 and HGT-014 do
not appear in the supplied module tables. Clarify those entries when their
future phases are scoped; do not invent their definitions.

### Requirements mapped to the prototype

| Source modules | Laptop prototype interpretation | Milestone |
| --- | --- | --- |
| PPE-001, PPE-004 | Helmet and high-visibility vest states on original-resolution worker crops | 3-5 |
| PPE-010, PPE-013 | Visibility-aware UNKNOWN state, confidence and recent observations per camera track; no cross-camera identity | 4-5 |
| BAR-001 | Tracked worker bottom-centre entering a configured polygon on a fixed camera; one event per entry episode | 7 |
| VEH-001 | Experimental image-space worker/vehicle risk; the source's <2 m criterion is deferred until calibration | 8 |
| IDT-004 | Counts of currently observed tracks within configured fixed-camera zones; no unique site-wide headcount claim | 7, 11 |
| EVD-001 | Snapshot, metadata and approximately 10 s before/10 s after a confirmed event | 9 |
| EVD-003 | Configurable event severity; separate confidence from severity | 5, 9-10 |
| EVD-011 | Persist acknowledgements and an action history in SQLite | 10-11 |
| PLT-001 | One RTSP camera with timeouts, reconnect handling and credential redaction | 6 |
| PLT-007, PLT-012 | Initial YAML polygons/rules and per-camera task switches; graphical no-code editing is future work | 6-7, 11 |
| EVD-007, EVD-008 | Basic observed counts and event summaries first; formal safety scores and rate definitions require later validation | 11-12 |

## MVP scope

Build in sequence: local MP4 person detection, temporary within-camera tracking,
helmet/vest training preparation, crop-based PPE inference and temporal state,
one RTSP feed, fixed-camera zone events, experimental vehicle risk, evidence,
SQLite and a basic Streamlit dashboard. Start with a pretrained nano-class
person detector. Train only the later PPE model through transfer learning.

Keep one selected feed active on the laptop. Cameras may have people, vehicles,
both, barricades or construction equipment; configure relevant tasks per camera.
Moving-camera support initially permits detection, tracking and possibly PPE.
Fixed polygons, calibrated distances, static-ground speed and fixed geometry
must be disabled while the camera is moving.

The prototype surfaces suspected safety events for review. UNKNOWN is a valid
PPE result. A detection miss alone is insufficient evidence of missing PPE.
Temporary track IDs are not employees, attendance or cross-camera identities.

## Hardware and resource constraints

| Item | Constraint or observed result |
| --- | --- |
| CPU | User reports AMD Ryzen 5 4600H |
| RAM | User reports 8 GB total system RAM |
| GPU | NVIDIA driver reports GeForce GTX 1650, 4096 MiB VRAM |
| OS | Environment check reports Windows 11, build 26200 |
| Interpreter | Existing `venv`, Python 3.14.5, 64-bit |
| Working tools | Windows, VS Code and the existing project structure |

CPU model and installed RAM were not independently verified: the sandbox denied
the optional Windows CIM hardware query. The user's stated limits remain the
design budget. GPU model and VRAM were independently read using `nvidia-smi`.

Start with one 1080p source, person inference around 640 pixels and PPE worker
crops around 384-416 pixels. A few analysed frames per second is acceptable.
The user's suggested person rate of 4-6 FPS and PPE rate of 2-4 observations
per relevant worker per second are starting targets, not benchmarks. Limit the
total crop workload as worker count grows, and measure achieved rates.

Retain the original frame until person crops are extracted. Avoid many models
loaded at once, unbounded queues and full-rate tiled inference. Use bounded
buffers and discard stale live frames. Process PPE selectively for relevant,
visible workers; optional tiling/ROIs must be limited to selected cameras,
distant regions and configured intervals. Evidence buffering must also fit RAM;
do not retain full-rate, uncompressed 1080p frames for every camera.

## Architecture and responsibilities

```text
Local MP4 / one RTSP camera
          |
     Frame reader ---------------- original-resolution frame
          |                                   |
 Person / vehicle detector                    |
          |                                   |
 Camera-local tracker                         |
          |                                   |
          +-- tracked person --> original-frame crop --> PPE observations
          |                                                 |
          |                                       Temporal PPE state
          |                                                 |
          +-- bottom-centre --> zone rules                   |
          +-- vehicle/person tracks --> proximity rules     |
                                    |                       |
                                    +-------- rule results -+
                                                 |
                                            Event engine
                                                 |
                                      Evidence + SQLite
                                                 |
                                        Streamlit dashboard
```

| Location | Responsibility and main input/output |
| --- | --- |
| `src/camera/` | Video/camera input to timestamped original frames and connection state |
| `src/detection/` | Frames/crops and model configuration to boxes, class scores and PPE observations |
| `src/tracking/` | Detections and timestamps to camera/session-local temporary tracks |
| `src/rules/` | Tracks, PPE history and configured geometry to suspected rule violations |
| `src/events/` | Rule transitions to deduplicated events and bounded evidence capture |
| `src/database/` | Events, evidence references and acknowledgements to SQLite records |
| `config/` | Camera tasks/env-variable references, rules/thresholds and zone polygons |
| `data/`, `models/`, `evidence/` | Local video/dataset inputs, model files and event evidence |
| `scripts/`, `tests/` | Milestone utilities and reproducible checks |

Keep model code independent of storage. A future event record includes
`event_id`, `camera_id`, `track_id`, `module_id`, `event_type`, timestamp,
confidence, severity, status, snapshot path and video path. Track references
also need camera/session context to avoid collisions across restarts. Future
SQLite tables may include cameras, events, event_evidence, zones and
acknowledgements. Store who acknowledged an event, when and any remarks.

Evidence should follow
`evidence/YYYY-MM-DD/event_<id>/{snapshot.jpg,event.mp4,metadata.json}`.
Short input clips, startup and disconnects can limit available pre/post footage;
record actual coverage instead of claiming a complete 20-second clip.

## Model and PPE strategy

Model A is a lightweight pretrained YOLO nano-class detector for people, then
supported vehicle classes. Select and record the exact checkpoint/version when
Milestone 1 is scoped and its compatibility is checked. Do not infer a usable
helmet model or validated performance from the master document's YOLO claims.
Construction vehicles absent from the chosen pretrained label set will need
separate evaluation and potentially later fine-tuning.

Model B is a separate helmet/vest detector, initially transferred from suitable
pretrained weights and a reviewed PPE dataset, then fine-tuned on labelled site
data. Convert Model A's boxes back to original-frame coordinates, add bounded
padding if useful, and clip crops to image boundaries before PPE analysis.
Upscaling a crop cannot recover details missing from the original image.

For each PPE item, keep PRESENT, MISSING and UNKNOWN. Head/torso visibility,
person height in original pixels, blur/occlusion and confidence determine
whether an observation is usable. A positive-only helmet/vest detector does
not by itself prove absence: before implementing MISSING, define and validate
the annotation and evidence policy for a clearly visible uncovered head/torso
or another explicitly validated absence signal. Non-detection stays UNKNOWN
without that evidence. No absent-detection-to-violation shortcut is allowed.

Starting examples from the user are person heights of 80 and 120 pixels, a
10-observation history and 7 required missing votes. These are experimental
configuration candidates, not scientific constants or active configuration.
Medium-size workers need stronger temporal confirmation. Define valid vote
counts, minimum observations, history expiry and elapsed-time limits together.
UNKNOWN must not count as MISSING or preserve a stale PRESENT state forever.

Use ByteTrack initially. Maintain histories per camera/session/track and expire
them after track loss. Configure tracking for sampled detection timestamps;
ByteTrack does not automatically observe or maintain accurate motion on frames
that were never supplied to it. Use configurable confirmation, episode state
and cooldown logic to suppress one-frame alarms and duplicate events.

## Training and dataset strategy

Training begins later, after recorded-video detection and tracking work:

1. Review dataset permissions, label quality, class meanings and CCTV relevance.
2. Extract frames at a configurable seconds interval, not every adjacent frame.
3. Include near/medium/far people, front/back/side views, partial occlusion,
   helmet/vest colours, sunlight, shadows, rain, crowds, gates and crane areas.
4. Keep training, validation and test sets separate by clips, time periods or
   cameras before deriving person crops. Keep related crops/frames in one split.
5. Fine-tune a nano-class pretrained model; do not start from random weights.
   Match crop-based training/validation to the intended PPE inference inputs.
6. Begin conservatively with `imgsz=640`, `batch=2`, `device=0`, `workers=2`,
   `cache=false` only after GPU checks pass. On CUDA out-of-memory, try batch 1
   first. Do not blindly increase image size. Tune based on measured RAM/VRAM.
7. Record dataset versions, splits, classes, configuration and seeds. Select
   weights using validation data, then evaluate the held-out test set and save
   the approved `best.pt` locally under `models/`.

Measure helmet/vest precision and recall overall and by original-frame person
height groups, plus UNKNOWN coverage and false/missed event counts. Choose
near/medium/far boundaries on site footage. Report sample counts, visibility,
camera/time split and limits. Do not invent accuracy or tune repeatedly on the
test set. Nightly automatic retraining is outside the MVP.

## Development milestones and completion criteria

Only Milestone 0 is authorized for this execution. Each later milestone needs
its necessary files, local checks, observed acceptance evidence, documentation
and a Git commit before proceeding. Unresolved gates must be reported honestly.

| Milestone | Deliverable | Observable acceptance check |
| --- | --- | --- |
| 0 - Audit/environment | Source mapping, this plan, AGENTS instructions and diagnostic script | Expected folders/files exist; existing venv is used; package and compute results plus blockers are reported; checker tests pass; no packages installed |
| 1 - Recorded video/person detection | OpenCV reader, one nano detector, person boxes/confidence/FPS and saved output | Process a supplied local MP4; reopen saved output and visually verify boxes, frame dimensions, end-of-file handling and measured throughput. No PPE/RTSP/tracking/database/dashboard |
| 2 - Tracking | ByteTrack and temporary ID overlay | Inspect an annotated clip for stable IDs on continuously visible workers; measure/report losses and switches; no employee/cross-camera identity claim |
| 3 - PPE training workspace | Interval extraction, dataset/PPE YAML, train/validate/predict utilities | Verify extraction times, group-disjoint splits and labels; complete a small authorized training/validation run with a loadable checkpoint and recorded memory use |
| 4 - PPE inference | Original-resolution crops, person size, helmet/vest observations and three states | Inspect labelled near/medium/far crops and overlays; verify PRESENT/UNKNOWN and evidence-supported MISSING; small/occluded observations must become UNKNOWN |
| 5 - Temporal state/events | Recent per-track history, configurable voting and suspected PPE events | Replay present/present/unknown/present and get no violation; sustained valid missing evidence yields one event per episode; verify expiry and independent tracks |
| 6 - One RTSP feed | Env-based URL lookup, timeout/reconnect and online/offline state | Connect one authorized camera, disconnect/recover and verify no unbounded retry/memory growth or secrets in output; moving-camera geometry guard is enforced |
| 7 - Restricted zones | YAML polygon, bottom-centre containment and zone counts | A tracked crossing generates exactly one BAR-001 episode event; exit/re-entry creates a new one; boundary and moving-camera cases are tested |
| 8 - Vehicle risk | Supported vehicle labels, tracks and image-space SAFE/WARNING/DANGER | Replay annotated risk examples; confirm configurable transitions and labels, with no uncalibrated metre/speed claims |
| 9 - Evidence | Snapshot, bounded pre/post video and metadata | Reopen saved images/video; verify event timestamp, available pre/post coverage, bounded memory and disk/disconnect failure handling |
| 10 - SQLite | Events/evidence references and acknowledgement history | Insert/read an event, persist across restart, acknowledge with actor/time/remarks and verify audit history and duplicate handling |
| 11 - Streamlit | Selected camera, observed counts, recent events, evidence and acknowledge controls | A reviewer finds an event, opens its evidence and records an acknowledgement visible after restart; only selected feeds are processed |
| 12 - Evaluation/optimisation | Site-specific report and operational checks | Publish measured per-size PPE results, false/missed events, FPS, RAM/VRAM and failure recovery on held-out clips; record reproducible settings and remaining limits |

Prototype completion requires a reproducible selected-camera pipeline through
events/evidence/storage/dashboard, verified failure handling and measured
results. Numeric accuracy acceptance thresholds must be agreed using labelled
site data; none have been established yet. Production capacity and certification
are not prototype completion criteria.

## Future scope retained from the master document

This register preserves all source module ranges. "Partial" means only the
limited interpretation above belongs in the MVP; remaining capabilities stay
in future phases.

| Category and source IDs | Retained capabilities and deferred work |
| --- | --- |
| PPE, PPE-001..014 (14) | Partial helmet/vest, occlusion and confidence; later fit/chin straps, goggles, boots, generic/welding/electrical gloves, harness presence, worker compliance trends, PPE removal and role/zone no-code PPE rules |
| Height, HGT-001..005 (5) | Calibrated/BIM elevation, anchored harness hook verification, guardrails, safe rope access and rescue-person presence |
| Excavation/confined space, EXC-001..004 (4) | Excavation barricades, buddy/lone-worker checks, excavator swing risks and trench-collapse early signs |
| Vehicles, VEH-001..004 (4) | Partial image-space person/vehicle risk; calibrated <2 m proximity, vehicle/vehicle near misses, >10 km/h speed and collision prediction |
| Barricades, BAR-001..003 (3) | Polygon entry in MVP; barricade tampering/removal and missing barricades around holes later |
| Fire/electrical/gas, FIR-001..008 (8) | Fire, smoke, cylinder storage, welding near flammables, missing/blocked extinguishers, exposed cable joints, water near electrical panels and smoking detection |
| Ergonomics/fatigue/lone worker, ERG-001..006 (6) | Lifting posture/frequency, overreaching, yawning/eye closure, long idle sitting and prolonged lack of movement; observed posture alone must not be treated as a medical diagnosis |
| Identity/hours/productivity, IDT-001..007 (7) | Cross-camera Re-ID, face attendance, unique site headcount, zone counts, productive/idle hours, overtime and SOS gestures; MVP has only camera-local counts/IDs |
| Evidence/reporting, EVD-001..011 (11) | Partial evidence/severity/audit; validated near-miss definitions, repeat violators with established identity, incident PDFs, WhatsApp/SMS/email/speaker alerts, safety scores, leading/lagging indicators and cross-site comparisons |
| Platform, PLT-001..014 (14) | Partial RTSP/config switches; bandwidth relay, Azure/AWS cloud compute, 3D fusion, EHS chat, risk forecasting, graphical rule builder, privacy/face masking, offline store-and-sync, mobile app, local-language voice alerts, false-alert feedback/retraining and edge/cloud hybrid deployment |

Supporting 80-120 cameras requires a separately sized production architecture.
Exact metric distances, work heights, speeds and 3D claims require calibration
and suitable ground truth. Face recognition, identity histories, compliance
claims and cloud services require their own scoped decisions. Do not infer
"saved lives" from a proximity event counter.

## Risks and mitigations

| Risk | Response and verification |
| --- | --- |
| CPU-only PyTorch despite visible NVIDIA GPU | Report separate driver and compute results. Resolve a compatible CUDA-enabled torch/torchvision pair only after explicit installation authorization, then rerun actual CUDA checks |
| Python/package/CUDA compatibility | Keep the existing venv. Verify official Windows/Python/driver/GPU support before a proposed change; never recreate the environment automatically |
| Small/distant or hidden PPE | Preserve source resolution, crop workers, use visibility/size gates and UNKNOWN, then evaluate by size and scene |
| No proof of missing PPE | Define supervised absence evidence and annotation rules before enabling missing-PPE violations |
| Tracking loss and stale state | Camera/session IDs, timed history expiry, representative occlusion tests and per-episode deduplication |
| Moving camera changes geometry | Disable static zones, metric proximity and speed; check the configuration guard |
| RAM/VRAM exhaustion and lag | One feed, nano models, bounded queues/crops/buffers, small batches and measured resource budgets |
| Dataset leakage/domain shift | Group splits before crops; diverse real CCTV labels; held-out near/medium/far evaluation |
| Camera/codec/disk failure | Test end-of-file, corrupt inputs, timeouts, reconnects, writable outputs and playable evidence in their milestones |
| Secret exposure | `.env` references only, ignored credentials, redacted logs; the M0 script never loads `.env` |
| Overstated accuracy/safety capability | Report measured evidence and limitations; source marketing claims remain unverified |

## Milestone 0 audit and reproducible checks

Audit date: 2026-09-09. The initial repository contained the seven starter files,
the requested directories and `venv`; there was no Git repository. Requirements
were reviewed and left unchanged: `ultralytics`, `opencv-python`,
`python-dotenv`, `pyyaml`. PyTorch and torchvision already exist as installed
dependencies. No installation or model download was performed.

| Check | Actual result |
| --- | --- |
| Python / interpreter | 3.14.5; project `venv\Scripts\python.exe` |
| Expected starter structure | PASS |
| PyTorch | 2.14.0+cpu; import and small CPU calculation PASS |
| torchvision | 0.29.0+cpu; import PASS |
| OpenCV | Runtime 5.0.0; distribution 5.0.0.93; import PASS |
| Ultralytics | 8.4.144; import and lazy YOLO class resolution PASS; no model constructed |
| python-dotenv / PyYAML | 1.2.3 / 6.0.3; imports PASS; `.env` not loaded |
| CUDA available / PyTorch CUDA build | False / None |
| GPU accessible through PyTorch | No; GPU calculation could not run |
| NVIDIA driver | GTX 1650; 4096 MiB VRAM; driver 610.47 |
| Dependency consistency | `pip check`: No broken requirements found |
| Checker tests | 11 unittest tests passed, including missing packages, wrong interpreter, CUDA calculation failure, timeout and secret-safe errors |
| Preservation checks | The 26,355 venv files retained the same path/size/modification-time fingerprint; `.env`, `.gitignore`, requirements and the three starter YAML files retained their SHA-256 hashes |
| Readiness | Exit code 2: CPU checks passed; GPU environment not ready |

The first full checker invocation hit a sandbox PermissionError during its
temporary-settings/import lifecycle. An approved diagnostic run outside the
sandbox completed with the results above. This sandbox failure was not evidence
of a broken ML package. The checker confines Ultralytics/matplotlib settings to
temporary storage, disables Ultralytics online checks/automatic installation,
suppresses raw third-party output and disables Python bytecode writes.

Run in PowerShell from the project root; activation is optional because these
commands explicitly select the existing interpreter:

```powershell
.\venv\Scripts\python.exe -B scripts/check_environment.py
$LASTEXITCODE
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B -m pip check
```

The checker also supports `--json`. Exit 0 means GPU environment checks passed;
exit 1 means prerequisite/check failure; exit 2 means CPU checks passed but GPU
readiness is blocked. A nonzero result must not be described as GPU success.
It does not test video codecs, model weights, full inference, accuracy or FPS.

### Files and how to use them

| File | Purpose, input and output | Run/check |
| --- | --- | --- |
| `PROJECT_PLAN.md` | Source document plus user constraints become this scoped roadmap and measured audit | Read in Markdown preview; compare milestone acceptance against actual results |
| `AGENTS.md` | Persistent engineering rules guide future repository work | Read before editing; review changes against its scope/security/testing rules |
| `scripts/check_environment.py` | Installed interpreter/packages and driver produce a console/JSON diagnostic plus readiness exit code | Run the checker command above; use the unit tests to check failure handling |
| `tests/test_check_environment.py` | Controlled dependency/GPU failure fixtures produce unittest pass/fail results | Run unittest above; these mocks do not substitute for the real environment run |
| `README.md` | Quick project orientation and links to the plan/check commands | Open the preview and follow commands from the project root |

Common errors: "file not found" usually means the command was run outside the
project root; "Using existing project venv: False" means the wrong interpreter;
PackageNotFoundError means a required distribution is missing; ImportError or
OSError may indicate incompatible packages/native libraries; a driver timeout
or missing `nvidia-smi` means driver information could not be read. Error types
are reported without raw messages that could leak secrets. Diagnose the named
component before changing dependencies; do not dump `.env` or the environment.

### Gate before Milestone 1

The laptop can import the CPU inference stack. It is **not yet ready for the
planned GPU-backed Milestone 1** because the installed torch/torchvision builds
are CPU-only. Do not downgrade Python or create a replacement venv based on that
finding. After explicit authorization, verify a supported CUDA-enabled wheel
pair for this Windows/Python/GPU/driver combination, update the existing venv
and rerun the diagnostic, tests and `pip check`. A deliberately CPU-only first
milestone is an alternative only if the user chooses that scope and its speed
limits are recorded. No package-install command is prescribed before that
compatibility check.

Git was initialized for this audited Milestone 0 checkpoint. The unresolved
GPU gate remains documented when the audit is committed. Verify its commit
with `git log -1 --oneline` and workspace state with `git status --short`.
A representative local MP4 is also needed for Milestone 1's observable video
test. Empty starter directories exist locally but Git does not track empty
directories; the environment checker will identify missing folders in a clone.

What to learn: a package being installed, a package importing, a driver seeing
the GPU and PyTorch executing GPU calculations are separate checks. A passing
environment audit does not establish detection quality.

Technical references consulted for the environment checks:
[PyTorch local installation and verification](https://pytorch.org/get-started/locally/)
and [Ultralytics installation](https://docs.ultralytics.com/quickstart/).
Use current official guidance when dependencies are changed; observed local
versions above describe this audit, not a recommended version combination.
