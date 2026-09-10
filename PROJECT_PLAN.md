# SafetyShield project plan

SafetyShield is a construction/factory computer vision safety monitoring
prototype. The first goal is a tested pipeline on one laptop and one video or
camera, followed by helmet/vest analysis, safety events, evidence and a simple
dashboard. Each milestone must demonstrate an observable result before the next
one begins.

Current stage: **Milestone 2 COMPLETE after manual prototype visual acceptance.**
The selected configurable person detector remains `yolo26n.pt`, `imgsz=960`,
confidence 0.20 on CUDA device 0. Selected configurable ByteTrack defaults are
high 0.20, low 0.10, new 0.20, buffer 45, match 0.80 and score fusion enabled.
The existing Python 3.14.5 environment now uses CUDA-enabled PyTorch and
torchvision. Actual CUDA matrix computation, the environment checker (exit 0),
all Milestone 0 unit tests, dependency consistency and Ultralytics import
passed. Milestone 1 measured real CUDA person inference and recorded-video
throughput, followed by manual prototype visual comparison. Formal labelled
accuracy evaluation remains deferred to Milestone 12, and no model training has
been performed.

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

Milestone 1 was explicitly authorized after Milestone 0 completed. Each later
milestone needs separate authorization, its necessary files, local checks,
observed acceptance evidence, documentation and a Git commit before proceeding.
Unresolved gates must be reported honestly.

| Milestone | Deliverable | Observable acceptance check |
| --- | --- | --- |
| 0 - Audit/environment | Source mapping, this plan, AGENTS instructions and diagnostic script; authorized CUDA package replacement | COMPLETE: existing venv retained; only torch/torchvision replaced; actual CUDA matrix computation passes; checker exit 0, all 11 tests, pip check and Ultralytics import pass |
| 1 - Recorded video/person detection | OpenCV reader, one nano detector, person boxes/confidence/FPS and saved output | **COMPLETE:** supplied MP4 processed on CUDA; output reopened with matching dimensions/frame count; throughput measured; controlled 640/0.25 versus 960/0.20 comparison manually accepted for the prototype. Formal labelled accuracy evaluation remains in Milestone 12. No PPE/RTSP/tracking/database/dashboard |
| 2 - Tracking | ByteTrack and temporary ID overlay | **COMPLETE:** Candidate B completed the full clip and was manually accepted from matched A/B contact sheets. This is prototype visual acceptance, not labelled tracking evaluation; no employee/cross-camera identity claim |
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
| CPU-only PyTorch despite visible NVIDIA GPU | Resolved in Milestone 0 through the authorized CUDA-enabled pair and real GPU execution. Recheck driver, imports and actual CUDA computation after future dependency changes |
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

## Milestone 0 audit history and reproducible checks

### Previous state

Initial audit date: 2026-09-09. The initial repository contained the seven starter files,
the requested directories and `venv`; there was no Git repository. Requirements
were reviewed and left unchanged: `ultralytics`, `opencv-python`,
`python-dotenv`, `pyyaml`. PyTorch and torchvision already exist as installed
dependencies. No installation or model download was performed during that
initial audit. Its CPU-only results are retained below as historical findings.

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

### Resolved state

Completion verification date: 2026-09-09. The user explicitly authorized
replacing only torch and torchvision in the existing project-root venv.
Python 3.14.5, all other installed package versions and `.env` were preserved.
The pre-install `pip check` passed and Git status was clean. The NVIDIA driver
reported GTX 1650, 4096 MiB VRAM, driver/KMD 610.47 and CUDA UMD 13.3.

The exact Windows CPython 3.14 wheels from the official CUDA 13.0 channel were
downloaded through `download.pytorch.org` and verified against the SHA-256
hashes published in the official index before removing the CPU packages. This
used the primary host because the index's alternate host had previously
returned HTTP 403. The replacement followed uninstall then install, using the
verified local wheels with `--no-index --no-deps --no-cache-dir --no-compile`.
Both operations succeeded; no other dependency was installed or upgraded.

| Completion check | Actual measured result |
| --- | --- |
| Python / interpreter | 3.14.5, 64-bit; existing `venv\Scripts\python.exe` |
| torch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| `torch.version.cuda` | 13.0 |
| `torch.cuda.is_available()` | True |
| CUDA device count | 1 |
| `torch.cuda.get_device_name(0)` | NVIDIA GeForce GTX 1650 |
| GPU VRAM | 4.0 GiB / 4096 MiB |
| Real CUDA computation | PASS: two 32x32 all-ones tensors on CUDA multiplied to a CUDA result containing 32.0 in every entry; synchronization succeeded |
| GPU memory after matrix check | 8.13671875 MiB allocated; 22.0 MiB reserved; these are allocator readings for this small check, not a model memory benchmark |
| torchvision CUDA operator | PASS: NMS on two identical synthetic boxes returned `[0]` on `cuda:0`; no detection model was used |
| `scripts/check_environment.py` | GPU ENVIRONMENT READY; actual `$LASTEXITCODE` = 0 |
| Unit tests | All 11 existing unittest tests passed unchanged |
| `pip check` | No broken requirements found |
| Ultralytics import | PASS; version 8.4.144; no YOLO model instantiated |
| Installed-package comparison | Only torch and torchvision changed; torchaudio remains absent |
| `.env` preservation | SHA-256 matches the pre-install snapshot; contents were not displayed or modified |

The original CPU-only torch/torchvision state had readiness exit 2. The resolved
state has CUDA-enabled builds and verified real CUDA execution with readiness
exit 0. No new venv, Python downgrade, standalone CUDA Toolkit, model download
or Milestone 1 implementation was performed.

Download files and the package inventory were stored outside the repository in
`%TEMP%\SafetyShield-cuda-recovery-2.14.0`. Ultralytics import settings were
confined there with online checks and automatic installation disabled. The
CUDA wheels, inventory and temporary library settings remain there; CPU
recovery wheels were not downloaded. Nothing from this temporary directory
or from venv belongs in the documentation commit.

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

The GPU environment gate is **resolved** by the completion checks above.
Milestone 0 is complete. At this historical checkpoint, Milestone 1 had not yet
started and still required authorization plus a representative recorded factory
MP4; it was later completed as documented below. GPU environment readiness by
itself did not establish person-detection accuracy, codec support or real-video
throughput.

Git was initialized for the initial audit checkpoint `2ab28dd`, which preserves
the former GPU blocker. The completion checkpoint contains only documentation
updates in `PROJECT_PLAN.md`, `AGENTS.md` and `README.md`; installed packages
remain ignored. Verify the latest commit with `git log -1 --oneline` and
workspace state with `git status --short`.
Empty starter directories exist locally but Git does not track empty
directories; the environment checker will identify missing folders in a clone.

What to learn: a package being installed, a package importing, a driver seeing
the GPU and PyTorch executing GPU calculations are separate checks. A passing
environment audit does not establish detection quality.

## Milestone 1 recorded-video person detection

Automated acceptance run date: 2026-09-10. The official COCO-pretrained
`yolo26n.pt` nano detection checkpoint was selected because it is the current
lightweight Ultralytics detection variant and fits the laptop's 4 GB VRAM
constraint better than larger variants. The threshold below is a configurable
starting point, not an optimized or scientifically validated value.

| Check | Actual observed result |
| --- | --- |
| Input video | `data/raw_videos/cam_good_test.mp4` |
| Source metadata | 1612x904; 30.000 FPS; 2,965 frames; approximately 98.833 seconds |
| Model | Ultralytics YOLO26 nano detection, checkpoint `yolo26n.pt` |
| Inference configuration | `imgsz=640`; confidence 0.25; person class 0 only; CUDA device 0 |
| CUDA device | NVIDIA GeForce GTX 1650; CUDA used for the real run |
| Processing | 2,965 frames in 141.261 seconds; 20.989 average end-to-end FPS |
| Model timing | 17.923 ms/frame average inference time reported by Ultralytics |
| Accepted detections | 2,645 person detections across the clip; this is not an accuracy measurement |
| GPU allocation | 58.401 MiB peak allocated according to `torch.cuda.max_memory_allocated()`; not total system GPU usage |
| Output | `outputs/cam_good_test_person_detected.mp4` |
| Output reopen | PASS: opened, readable first frame, 1612x904, 30.0 FPS, 2,965 frames; dimensions match input |
| Automated tests | PASS: all 15 unit tests (11 existing plus 4 Milestone 1 tests) |
| Dependency consistency | PASS: `pip check` reported no broken requirements |
| Initial visual review | Close workers detected well; some medium/distant workers missed, leading to the controlled tuning experiment below |

The reader validates file existence, OpenCV opening, dimensions and FPS; exposes
frame count/duration; distinguishes expected end-of-file from an early decode
failure when frame count metadata is available; and releases the capture. The
detector loads one checkpoint, requires CUDA, restricts inference to COCO person
class 0 and returns original-frame clipped coordinates. The runner draws small
labels, writes at the original size, records actual processing measurements and
reopens the generated MP4.

Run from PowerShell in the project root:

```powershell
.\venv\Scripts\python.exe -B scripts\run_person_detection.py --input data\raw_videos\cam_good_test.mp4
.\venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\venv\Scripts\python.exe -B -m pip check
```

| File | Purpose, input and output |
| --- | --- |
| `src/camera/video_reader.py` | Local MP4 path to validated metadata and original-resolution decoded frames |
| `src/detection/person_detector.py` | Original BGR frame plus model settings to clipped person detections |
| `scripts/run_person_detection.py` | Input video and detections to labelled output MP4, timings, memory reading and reopen check |
| `tests/test_person_detection.py` | Synthetic video/structures to reader, clipping and output-directory checks; it does not test detection accuracy |

Common errors: a missing input means the supplied MP4 is not at the requested
path; an invalid FPS or dimension indicates unusable video metadata; an early
decode error may indicate a corrupt/truncated clip or codec problem; a CUDA
error means device 0 must be diagnosed rather than silently using CPU; output
writer failure may mean the output directory is unwritable or the MP4 codec is
unavailable. Model download is only needed when the ignored checkpoint is not
already local.

Do not infer accuracy from the successful run or the detection count. The user
must inspect close, medium, distant, moving and partially hidden workers; false
detections and missed obvious people; bounding-box alignment; confidence labels;
and playback quality. Stop at Milestone 1 and wait for that review.

### Controlled detection tuning experiment

Experiment date: 2026-09-10. Manual review of the first Milestone 1 output found
good close-worker detection but missed some medium/distant workers. The same
input and `yolo26n.pt` checkpoint were therefore run twice to isolate the
combined effect of inference size and confidence threshold. The original
`outputs/cam_good_test_person_detected.mp4` was preserved.

| Check | A: baseline | B: higher resolution/lower confidence |
| --- | --- | --- |
| Model / device | `yolo26n.pt`; CUDA 0, GTX 1650 | `yolo26n.pt`; CUDA 0, GTX 1650 |
| Inference size / confidence | 640 / 0.25 | 960 / 0.20 |
| Frames processed | 2,965 | 2,965 |
| Accepted person detections | 2,645 | 5,970 |
| Processing time | 130.130 seconds | 143.596 seconds |
| Average end-to-end FPS | 22.785 | 20.648 |
| Ultralytics-reported inference time | 17.069 ms/frame | 17.998 ms/frame |
| Peak PyTorch allocated GPU memory | 66.077 MiB | 98.711 MiB |
| Output video | `outputs/person_detection_640_conf025.mp4` | `outputs/person_detection_960_conf020.mp4` |
| Output reopen | PASS: 1612x904, 30 FPS, 2,965 frames, readable frame | PASS: 1612x904, 30 FPS, 2,965 frames, readable frame |

Compared with A, B recorded 3,325 more accepted detections (125.7% more),
2.137 fewer end-to-end FPS (9.4% lower), 13.466 seconds more processing time,
0.929 ms/frame more reported inference time and 32.634 MiB more peak PyTorch
allocated memory. More detections do not establish better quality because they
may include false positives. Detection-quality comparison was subsequently
accepted through the scoped manual review recorded below.

`scripts/extract_detection_comparisons.py` maps timestamps to frame indices
using the shared 30 FPS metadata, reads that same index from both outputs and
writes full-resolution side-by-side PNGs without using them as an automatic
accuracy benchmark. Matching comparisons were generated at 10, 30, 50, 70 and
80 seconds (frames 300, 900, 1500, 2100 and 2400) under
`outputs/comparison/`. The output directory remains ignored by Git.

```powershell
.\venv\Scripts\python.exe -B scripts\extract_detection_comparisons.py `
  --baseline outputs\person_detection_640_conf025.mp4 `
  --higher outputs\person_detection_960_conf020.mp4 `
  --output-dir outputs\comparison `
  --timestamps 10 30 50 70 80
```

All 16 unit tests passed after adding the timestamp/frame-index check, and
`pip check` reported no broken requirements.

### Milestone 1 manual visual acceptance and selected baseline

The user manually reviewed the matching 10, 30, 50, 70 and 80 second comparison
images. Configuration B (`yolo26n.pt`, `imgsz=960`, confidence 0.20) was selected
as the Milestone 1 baseline because it substantially improved prototype visual
detection of medium and distant workers in those samples:

- at 10 seconds, B recovered a distant worker missed by A;
- at 30 seconds, B recovered visible foreground and distant workers missed by A;
- at 50 seconds, both detected the main worker and B recovered a distant worker;
- at 70 seconds, B recovered additional valid visible workers; and
- at 80 seconds, A detected one nearby worker while B detected both nearby
  workers and an additional distant worker.

No obvious false-positive problem was observed in these five comparison frames.
This is a manual prototype visual evaluation, not a labelled benchmark. It does
not establish precision, recall, mAP or accuracy. Formal labelled accuracy
evaluation remains required in Milestone 12.

| Selected setting/result | Milestone 1 value |
| --- | --- |
| Checkpoint | `yolo26n.pt` |
| Person inference size | 960 |
| Person confidence threshold | 0.20 |
| Device | CUDA device 0, NVIDIA GeForce GTX 1650 4 GB |
| Measured tuning run | 2,965 frames; 5,970 accepted detections; 20.648 end-to-end FPS; 17.998 ms/frame reported inference; 98.711 MiB peak PyTorch allocation |
| Output resolution | 1612x904 |
| Status | **MILESTONE 1 COMPLETE** |

The selected inference size and confidence remain configurable prototype
defaults, not permanent scientific constants. The measured runtime values are
specific to this run and are not production guarantees. At this checkpoint,
Milestone 2 had not started and still required separate authorization.

## Milestone 2 camera-local person tracking

Automated acceptance run date: 2026-09-10. The existing Milestone 1 detector
feeds person detections into Ultralytics ByteTrack. The required `lap==0.5.13`
package was explicitly authorized and installed with `--no-deps`; no other
package was installed or upgraded. Torch remained `2.14.0+cu130`, torchvision
remained `0.29.0+cu130`, the PyTorch CUDA build remained 13.0, the environment
checker returned 0 and `pip check` reported no broken requirements.

| Check | Actual observed result |
| --- | --- |
| Tracker | Ultralytics ByteTrack through a separate `PersonTracker` abstraction |
| Tracker configuration | high threshold 0.25; low threshold 0.10; new-track threshold 0.25; buffer 30 frames; match threshold 0.80; score fusion enabled |
| Scope | Person class only; IDs local to `camera_id` plus `session_id`; no identity or cross-camera claim |
| Input | `data/raw_videos/cam_good_test.mp4`; 1612x904; 30 FPS; 2,965 frames |
| Detector | `yolo26n.pt`; `imgsz=960`; confidence 0.20; CUDA device 0 on NVIDIA GeForce GTX 1650 |
| Accepted person detections | 5,970 |
| Processing | 2,965 frames in 147.724 seconds; 20.071 average end-to-end FPS |
| Detector timing | 17.374 ms/frame average inference time reported by Ultralytics |
| GPU allocation | 98.711 MiB peak allocated according to `torch.cuda.max_memory_allocated()`; not total system GPU usage |
| Temporary track IDs | 57 IDs appeared in tracker output during this session; this is not a count of unique real people |
| Maximum active tracks | 5 simultaneously returned tracks |
| Output | `outputs/cam_good_test_tracked.mp4` |
| Output reopen | PASS: opened, readable first frame, 1612x904, 30 FPS, 2,965 frames; dimensions match input |
| Review samples | 10, 30, 50, 70 and 80 seconds under `outputs/tracking_samples/` |
| Automated tests | PASS: all 23 unit tests; mocked/synthetic tests do not prove real tracking quality |
| Initial visual tracking review | **NEEDS TUNING:** apparent fragmentation motivated the controlled experiment below |

`src/tracking/person_tracker.py` owns ByteTrack configuration and conversion
from person detections to `TrackedPerson` observations. Each observation carries
camera ID, session ID, temporary track ID, frame/time, confidence and clipped
original-frame coordinates, plus derived width, height and bottom-centre values
for later authorized milestones. A local mapping prevents separate tracker
instances from exposing shared ID numbering. Empty detections and invalid boxes
are handled without crashing.

Run the recorded-video tracker from the project root:

```powershell
.\venv\Scripts\python.exe -B scripts\run_person_tracking.py --input data\raw_videos\cam_good_test.mp4
```

The output labels use `Track N | confidence`. They identify only a temporary
track within the current camera/session. The 57 generated IDs cannot establish
the number of real people because a person may receive another ID after track
loss. No IDF1, MOTA, HOTA or ID-switch rate is reported without labelled
tracking ground truth.

Manual review must check continuous-person ID stability, unnecessary changes,
partial-occlusion recovery, swaps when people pass, distant-worker tracking,
persistent false detections, box alignment and the observed performance impact.
Do not start Milestone 3 until this visual review is complete and the next
milestone is explicitly authorized.

### Controlled ByteTrack tuning experiment

Experiment date: 2026-09-10. Initial manual review found generally aligned
boxes, separate temporary tracks for multiple people and tracking of distant
workers when detections were available, but also apparent fragmentation during
intermittent detections and occlusion. This is a manual prototype observation,
not a labelled tracking benchmark. No IDF1, MOTA, HOTA or ID-switch rate is
claimed.

The candidate aligns ByteTrack's high and new-track thresholds with the selected
0.20 detector confidence floor and experimentally extends retention from 30
frames (about 1.0 second at 30 FPS) to 45 frames (about 1.5 seconds). All other
detector and tracker settings remain identical. These tracker values are
configurable experiment settings, not validated constants.

| Check | A: baseline | B: candidate |
| --- | --- | --- |
| Detector | `yolo26n.pt`; `imgsz=960`; confidence 0.20; person only; CUDA 0 | Same |
| ByteTrack high / low / new threshold | 0.25 / 0.10 / 0.25 | 0.20 / 0.10 / 0.20 |
| Buffer / match / score fusion | 30 / 0.80 / enabled | 45 / 0.80 / enabled |
| Frames | 2,965 | 2,965 |
| Processing time | 147.724 seconds | 140.838 seconds |
| Average end-to-end FPS | 20.071 | 21.053 |
| Peak PyTorch allocated GPU memory | 98.711 MiB | 98.711 MiB |
| Temporary track IDs generated | 57 | 54 |
| Maximum simultaneous active tracks | 5 | 5 |
| Output | `outputs/cam_good_test_tracked.mp4` | `outputs/cam_good_test_tracked_candidate_b.mp4` |
| Output reopen | PASS: 1612x904, 30 FPS, 2,965 frames | PASS: 1612x904, 30 FPS, 2,965 frames, readable first frame |

The candidate produced three fewer temporary IDs, but that diagnostic does not
prove better continuity and is not a unique-person count. Run-to-run throughput
differences are measured values, not production guarantees. No optional
track-lifetime heuristic was added because rendered tracker output alone cannot
establish when two IDs belong to the same real person.

`scripts/extract_tracking_comparisons.py` reads the preserved baseline and
candidate annotated videos without rerunning detection or tracking. It creates
six ignored 6448x4740 JPEG sheets under
`outputs/tracking_tuning/comparison_sheets/`: two sheets per 10-second interval
for 5-15, 25-35 and 75-85 seconds. Each sheet preserves full-resolution video
tiles and labels A/B, absolute timestamp and frame index for ten matching
0.5-second samples.

All 26 unit tests passed outside the restrictive sandbox after the sandboxed
run encountered its known temporary-file permission limitation. `pip check`
reported no broken requirements. The environment checker returned 0 and
reconfirmed torch 2.14.0+cu130, torchvision 0.29.0+cu130, CUDA build 13.0 and
CUDA execution on the NVIDIA GeForce GTX 1650. No package was installed or
changed.

### Milestone 2 manual visual acceptance and selected baseline

The user manually reviewed the A/B contact sheets for 5-15, 25-35 and 75-85
seconds, sampled at approximately 0.5-second intervals. Bounding boxes were
generally aligned; continuously visible people had acceptable temporary-track
continuity for the prototype; distant people were tracked when usable person
detections existed; and multiple nearby people showed no obvious ID swap in the
reviewed samples. No obvious persistent false-track problem or meaningful visual
regression from A was observed.

Candidate B is therefore the selected Milestone 2 prototype configuration:
Ultralytics ByteTrack with high 0.20, low 0.10, new 0.20, buffer 45, match 0.80
and score fusion enabled. The detector remains `yolo26n.pt`, `imgsz=960`,
confidence 0.20, person class only on CUDA device 0. These are configurable
prototype defaults, not permanent scientific constants.

This was sampled manual prototype review, not labelled tracking ground truth.
It establishes no IDF1, MOTA, HOTA, ID-switch rate or tracking-accuracy
percentage. Formal labelled tracking evaluation remains future evaluation work,
and 54 temporary IDs must not be interpreted as 54 unique people.

Closure verification passed all 27 unit tests and `pip check`. The environment
checker returned 0 and reconfirmed CUDA execution on the GTX 1650 with torch
2.14.0+cu130, torchvision 0.29.0+cu130 and PyTorch CUDA build 13.0 unchanged.

**MILESTONE 2 STATUS: COMPLETE.** Do not start Milestone 3 until it is explicitly
authorized.

Technical references consulted for the environment checks:
[PyTorch local installation and verification](https://pytorch.org/get-started/locally/)
and [Ultralytics installation](https://docs.ultralytics.com/quickstart/).
Use current official guidance when dependencies are changed; observed local
versions above describe this audit, not a recommended version combination.
