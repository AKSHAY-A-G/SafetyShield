# SafetyShield project plan

SafetyShield is a construction/factory computer vision safety monitoring
prototype. The first goal is a tested pipeline on one laptop and one video or
camera, followed by helmet/vest analysis, safety events, evidence and a simple
dashboard. Each milestone must demonstrate an observable result before the next
one begins.

Current stage: **Milestone 6 COMPLETE.** Milestones 0-6
are complete. Milestone 7 has not started and requires separate authorization.
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
| 3 - Restricted zones | YAML polygon, bottom-centre containment and zone counts | **COMPLETE:** BAR-001 entry event and IDT-004 occupancy validated on `cam_good_test.mp4`; manual visual review accepted |
| 4 - Temporal safety rules | EXC-002 buddy zone and ERG-006 low-movement rules | **COMPLETE:** EXC-002 single-occupancy event and ERG-006 low-movement timing validated; manual visual review accepted |
| 5 - Event evidence | Raw/annotated snapshots, bounded clips, metadata, audit log | **COMPLETE:** 1612x904 snapshots, 10s clipped MP4, JSON metadata, and JSONL audit logging validated; manual visual review accepted |
| 6 - One RTSP feed | Env-based URL lookup, timeout/reconnect, latest-frame slot | **COMPLETE:** 120s live acceptance run on `live_cam_1` (2560x1440, ~10.2 FPS) passed; clean shutdown; manual visual review accepted |
| 7 - Basic dashboard and module controls | Streamlit dashboard, system status, module toggles, telemetry, evidence explorer | **IMPLEMENTED:** Offline tests passed (139/139); local Streamlit server verified; visual acceptance PENDING USER REVIEW |
| 8 - Next milestone | To be determined upon user authorization | NOT STARTED / requires separate authorization |

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

## Milestone 3 fixed-camera restricted-zone rules

Automated acceptance run date: 2026-09-11. The user explicitly authorized this
zone/rule milestone after accepting Milestone 2 and manually saved the prototype
polygon with `scripts/draw_zone.py`. This scoped authorization supersedes the
earlier roadmap ordering for this checkpoint only; no PPE or later module was
started.

The sole configured zone is camera `cam_good_test`, zone
`restricted_zone_1` / `Restricted Zone 1`, type `restricted`, enabled, at the
expected fixed-camera resolution 1612x904. The polygon is exactly
`[(471, 463), (653, 403), (619, 549), (439, 581)]`. There are no additional
fabricated zones. Inside confirmation is 0.20 seconds, outside confirmation is
0.20 seconds and per-track zone state expires after 2.0 seconds without an
observation. Polygon boundaries count as inside.

Each tracked person's bottom-centre point `((x1 + x2) // 2, y2)` determines
zone membership. IDT-004 is the count of currently active, confirmed-inside
temporary tracks for this camera and session only. It is not a unique-person,
employee, attendance or site-wide worker count.

BAR-001 requires a confirmed OUTSIDE state followed by a confirmed INSIDE
state. A track first observed inside can contribute to occupancy after inside
confirmation but does not generate an entry event. Remaining inside does not
repeat the event. A confirmed exit followed by confirmed re-entry can generate
another event. Both initial outside qualification and subsequent exits use the
configured outside debounce. Track fragmentation can affect event totals, and
temporary ByteTrack IDs do not establish identity.

The full-video run preserved the selected detector (`yolo26n.pt`, `imgsz=960`,
confidence 0.20, person class only, CUDA device 0) and selected ByteTrack values
(high 0.20, low 0.10, new 0.20, buffer 45, match 0.80, score fusion enabled).

| Check | Actual observed result |
| --- | --- |
| Input | `data/raw_videos/cam_good_test.mp4`; 1612x904; 30 FPS; 2,965 frames |
| Processing | 2,965 frames in 154.998 seconds; 19.129 average end-to-end FPS |
| BAR-001 events | 1 entry event: Track 9 at 38.200 seconds / frame 1,146; not a unique-person count |
| Maximum IDT-004 occupancy | 1 active confirmed-inside temporary track |
| GPU allocation | 98.711 MiB peak allocated according to `torch.cuda.max_memory_allocated()`; not total system GPU use |
| Output | `outputs/cam_good_test_zones.mp4`; generated artifact remains ignored by Git |
| Output reopen | PASS: readable frame, 1612x904, 30 FPS, 2,965 frames, dimensions matched |
| Review material | Default frames in `outputs/zone_samples/`; event-focused frames at 35.0, 37.5, 38.2, 38.7, 40.0, 42.0 and 45.0 seconds in `outputs/zone_event_review/`; ignored by Git |
| Unit tests | PASS: all 47 tests; synthetic/mocked tests do not prove real CV quality |
| Dependency check | PASS: `pip check` found no broken requirements |
| Environment check | PASS, exit 0: torch 2.14.0+cu130, torchvision 0.29.0+cu130, CUDA calculation on NVIDIA GeForce GTX 1650 |

The annotated output overlays the polygon and name, current occupancy, person
boxes, temporary Track IDs, bottom-centre markers, green inside boxes and a
one-second BAR-001 entry banner. The banner's presence on multiple rendered
frames is display duration only; the rule engine generated one event.

### Milestone 3 manual visual acceptance

The user manually reviewed the event-focused frames at 35.0, 37.5, 38.2, 38.7,
40.0, 42.0 and 45.0 seconds and accepted the behavior for this prototype. Track
9 was outside with occupancy 0 at 35.0 seconds, approaching/crossing while
confirmation kept occupancy at 0 at 37.5 seconds, and confirmed inside with
occupancy 1 and the BAR-001 entry indication at 38.2 seconds. It remained
inside with occupancy 1 at 38.7 and 40.0 seconds without another event. At 42.0
seconds it approached/left the boundary consistently with exit debounce, and by
45.0 seconds it was clearly outside with occupancy 0. The yellow bottom-centre
marker behaved consistently with polygon membership, and no obvious edge-jitter
problem appeared in the reviewed sequence.

This is manual prototype visual acceptance, not labelled accuracy evaluation.
Track IDs remain temporary and camera/session-local; the BAR-001 event count is
not a unique-person count; IDT-004 reports current camera-local occupancy rather
than unique site workers; and tracker fragmentation can affect downstream
events. The polygon is a prototype/test zone, fixed polygon rules require a
fixed camera viewpoint, and formal evaluation remains future work.

**MILESTONE 3 STATUS: COMPLETE.**

**ZONE/RULE VISUAL ACCEPTANCE: PASS (manual prototype review).** Do not start
Milestone 4 until the user explicitly authorizes it.

## Milestone 4 easy-first temporal safety rules

Automated acceptance run date: 2026-09-11. The user explicitly authorized only
EXC-002 and ERG-006 after completing Milestone 3. This scoped sequencing
supersedes the earlier roadmap ordering for this checkpoint. No new model,
training, PPE, RTSP, evidence persistence, database or later milestone work was
added.

`src/rules/temporal.py` keeps temporal state separate from detection, tracking
and polygon geometry. `config/rules.yaml` supplies all important thresholds.
The recorded-video runner reuses the accepted detector, ByteTrack configuration
and Milestone 3 zone engine without changing them.

### EXC-002 configured buddy-required zone

EXC-002 represents exactly one confirmed current track in a configured
buddy-required monitoring zone. It is a state rule, so a system starting with
one person already inside can generate an event after confirmation; unlike
BAR-001 it does not require an observed crossing. The prototype confirmation
threshold is 3.0 seconds and reset confirmation is 1.0 second. Both values are
configurable experimental defaults, not regulatory, certified or production-
validated values. One event is emitted per continuous single-occupancy episode;
confirmed occupancy 0 or 2+ clears the episode, after which a later confirmed
single-occupancy episode may produce another event.

The accepted `restricted_zone_1` polygon is reused only to exercise the generic
rule on the recorded clip. It is a manually configured prototype/test zone, not
a validated confined space. SafetyShield does not automatically recognize a
confined space in this milestone.

### ERG-006 prolonged low movement

ERG-006 measures the image-space bottom-centre trajectory of each temporary
camera/session-local track while it is confirmed inside an applicable zone. The
stationary anchor is reset only when displacement exceeds
`max(5 pixels, 0.10 * person bounding-box height)`, preventing small tracker
jitter from continuously resetting the timer while remaining scale-aware. No
pixel-to-metre conversion, pose estimation, optical flow or additional AI model
is used.

The configured no-movement duration remains 600 seconds, matching the master
requirement rather than being shortened for the approximately 99-second test
clip. State expires after 2.0 seconds without an applicable track observation.
Track loss/expiry or a new Track ID starts fresh history; identity is not
inferred across IDs. One event is emitted per prolonged low-movement episode,
and meaningful movement resets the episode. This is only a prolonged
low-movement safety alert for human interpretation, not medical-emergency,
fatigue, unconsciousness or diagnosis detection.

| Check | Actual observed result |
| --- | --- |
| Input | `data/raw_videos/cam_good_test.mp4`; 1612x904; 30 FPS; 2,965 frames |
| Locked pipeline | `yolo26n.pt`; `imgsz=960`; confidence 0.20; person only; CUDA 0; ByteTrack high/low/new 0.20/0.10/0.20, buffer 45, match 0.80, score fusion enabled; saved zone unchanged |
| Processing | 2,965 frames in 154.350 seconds; 19.210 average end-to-end FPS |
| Performance comparison | 0.081 FPS above the Milestone 3 run (about 0.4%); no substantial temporal-rule slowdown observed |
| BAR-001 regression | PASS: 1 entry event, Track 9 at 38.200 seconds/frame 1,146 |
| IDT-004 regression | PASS: maximum confirmed camera-local occupancy 1 |
| EXC-002 | 1 event, Track 9 in `restricted_zone_1` at 41.200 seconds/frame 1,236 |
| ERG-006 | 0 events; expectedly possible because the clip is shorter than the configured 600-second requirement |
| GPU allocation | 98.711 MiB peak allocated according to `torch.cuda.max_memory_allocated()`; not total system GPU use |
| Output | `outputs/cam_good_test_temporal_rules.mp4`; generated artifact remains ignored by Git |
| Output reopen | PASS: readable frame, 1612x904, 30 FPS, 2,965 frames, dimensions matched |
| Review material | Frames at 35.0, 38.0, 39.2, 40.0, 41.2, 41.9, 42.0, 42.3 and 45.0 seconds in `outputs/temporal_rule_samples/`; ignored by Git |
| Unit tests | PASS: all 67 tests, including 20 focused temporal tests and existing BAR-001/IDT-004 regression tests |
| Dependency check | PASS: `pip check` found no broken requirements; no dependency was installed or changed |
| Environment check | PASS, exit 0: torch 2.14.0+cu130, torchvision 0.29.0+cu130 and a CUDA calculation on the NVIDIA GeForce GTX 1650 |

### Milestone 4 manual visual acceptance

The user manually reviewed the temporal-rule frames and accepted the behavior
for this prototype. The buddy-required test zone was clear with Track 9 outside
at 35.0 seconds. At approximately 38 seconds Track 9 approached/entered without
an immediate EXC-002 alert. Exactly one confirmed occupant was present at 39.2
seconds, with confirmation near 1.0 second and the existing BAR-001 indication;
the low-movement timer remained short because the person was moving. At 40.0
seconds confirmation had progressed to about 1.8 seconds without a premature
event.

At 41.2 seconds the configured 3.0-second confirmation was satisfied and the
EXC-002 LONE WORKER indication appeared; automated execution recorded one event
at frame 1,236. EXC-002 entered reset-pending at 41.9 seconds. The condition
returned at 42.0-42.3 seconds before the configured 1.0-second reset completed,
so the same episode remained active without a new event. At 45.0 seconds the
person was outside and the condition was clear. Multiple frames displaying
LONE WORKER are presentation duration/state, not multiple events.

The displayed ERG-006 low-movement duration remained short or reset while the
person moved. No ERG-006 event was generated, which is consistent with the
600-second configured threshold and approximately 99-second video. The runtime
threshold was not reduced to force a demonstration.

| File | Purpose, input and output |
| --- | --- |
| `config/rules.yaml` | Runtime zone applicability, confirmation/reset, movement and expiry thresholds |
| `src/rules/temporal.py` | Zone counts and tracked observations to in-memory EXC-002/ERG-006 states and events |
| `scripts/run_temporal_rules.py` | Local MP4 through the existing detection/tracking/zone pipeline to a labelled ignored MP4 and review frames |
| `tests/test_temporal_rules.py` | Simulated timestamps, occupancies and tracks to temporal semantics checks; not real safety accuracy evaluation |

Common errors: an unknown zone ID means `config/rules.yaml` does not match an
enabled configured zone; an invalid threshold reports a configuration error;
missing or corrupt video, CUDA and codec failures retain the existing runner
diagnostics. Fragmented tracks can reset ERG-006 history and can affect temporal
events. Formal labelled evaluation remains future work.

**MILESTONE 4 STATUS: COMPLETE.**

**TEMPORAL RULE VISUAL ACCEPTANCE: PASS (manual prototype review).** This is not
formal accuracy evaluation.

## Milestone 5: Event records, evidence capture, and prototype audit trail

Milestone 5 converts transient in-memory safety rule events into structured,
persistent evidence records and an append-only software audit log. It addresses
source requirements `EVD-001` (evidence capture) and `EVD-011` (prototype audit
trail). No new AI model or dataset training was required; all existing detector,
tracker, zone, and temporal rule parameters remain locked.

### Common event representation and schema

Safety events emitted by the rule engines are standardized into a common
`SafetyEvent` record (`src/events/models.py`), decoupled from computer vision
inference:

- `event_id`: Unique string identifier generated per event using a safe
  collision-resistant prefix + random hex (`evt_<prefix>_<uuid_hex>`). It is
  completely independent of employee identity, credentials, or session IDs.
- `schema_version`: `"1.0"`.
- `camera_id`, `session_id`, `track_id`: Preserve camera-local context and
  temporary track identity.
- `module_id`, `event_type`: Identify rule origin (e.g. `BAR-001` /
  `restricted_zone_entry`, `EXC-002` / `buddy_required_single_person`, `ERG-006` /
  `prolonged_low_movement`).
- `source_timestamp_seconds`: Explicit source-video position in seconds.
- `frame_number`: Explicit source frame index.
- `zone_id`, `zone_name`: Applicable zone geometry metadata when relevant.
- `severity`: Standard prototype value `"unclassified"`. Milestone 5 does not
  implement automated severity classification (`EVD-003` is deferred).
- `status`: Standard initial value `"new"`. No acknowledgement workflow is
  implemented yet.
- `detection_confidence`, `stationary_duration_seconds`, `zone_occupancy`:
  Optional rule-specific indicators.
- `created_at_utc`, `processed_at_utc`: Distinct ISO 8601 UTC wall-clock
  timestamps, strictly separated from recorded-video `source_timestamp_seconds`.

### Evidence directory structure

All generated evidence is saved under the Git-ignored `evidence/` directory:

```text
evidence/
  <camera_id>/
    <session_id>/
      audit.jsonl
      event_<event_id>/
        snapshot_raw.jpg
        snapshot_annotated.jpg
        event_clip.mp4
        metadata.json
```

1. `snapshot_raw.jpg`: The original CCTV frame before SafetyShield overlays,
   retaining the native source resolution (1612x904).
2. `snapshot_annotated.jpg`: The same event frame containing bounding boxes,
   Track ID, zone polygon, and event banner at 1612x904.
3. `event_clip.mp4`: Short event-centered evidence clip extracted directly from
   the original recorded source MP4, avoiding in-memory ring buffers.
   Configurable prototype defaults are `pre_event_seconds = 5.0` and
   `post_event_seconds = 5.0` (~10.0-second total window). Clips are safely
   clamped to source video bounds `[0.0, duration]`. Every written clip is
   reopened and verified (file exists, readable frame, dimensions match, valid
   FPS, frame count > 0).
4. `metadata.json`: Contains complete event attributes and relative artifact
   paths (`raw_snapshot`, `annotated_snapshot`, `video_clip`), requested vs
   actual clip coverage (`requested_pre_seconds`, `requested_post_seconds`,
   `actual_clip_start_seconds`, `actual_clip_end_seconds`, `actual_pre_seconds`,
   `actual_post_seconds`, `clip_frame_count`), and resolution without exposing
   credentials or machine-specific absolute paths.
5. Attempting to save evidence to an existing event directory raises
   `FileExistsError` to prevent silent overwriting.

### Prototype audit trail (EVD-011)

An append-only software audit log is written to `audit.jsonl` using JSON Lines
(`src/events/audit.py`). Each line is an independent JSON object recording
lifecycle actions:
- `EVENT_CREATED`: Logged when a safety rule fires and an event record is formed.
- `EVIDENCE_SAVED`: Logged when snapshots, clip extraction, and metadata are
  successfully committed.
- `EVIDENCE_ERROR`: Logged if an evidence extraction or filesystem error occurs.

**Audit limitations**: This is a prototype software log for local traceability.
It is NOT tamper-proof, immutable, forensically certified, or
regulatory-compliant. Anyone with filesystem access can alter the file.

### Milestone 5 automated execution results

The full factory video `data/raw_videos/cam_good_test.mp4` (2,965 frames,
1612x904, 30 FPS, 98.833 s) was processed end-to-end via
`scripts/run_evidence_pipeline.py` with session ID `milestone5_test`.

| Check | Actual observed result |
| --- | --- |
| Input | `data/raw_videos/cam_good_test.mp4`; 1612x904; 30 FPS; 2,965 frames (98.833 s) |
| Locked pipeline | `yolo26n.pt`; `imgsz=960`; conf 0.20; CUDA 0; ByteTrack defaults; zones and temporal configs unchanged |
| Main pipeline processing | 2,965 frames in 113.645 seconds; 26.090 average FPS |
| Evidence generation time | 11.735 seconds for 2 complete evidence packages |
| Peak PyTorch GPU memory | 81.931 MiB allocated |
| BAR-001 event | 1 event: `bar_9_c3da8e260ac14b6a` at 38.200 s (frame 1,146, Track 9) |
| EXC-002 event | 1 event: `exc_002_9_16009694b29049b5` at 41.200 s (frame 1,236, Track 9) |
| ERG-006 events | 0 events (consistent with 600 s threshold on 98.8 s video; no fabricated evidence) |
| Maximum observed occupancy | 1 person in `restricted_zone_1` (IDT-004 regression PASS) |
| Raw snapshots saved | 2 images (1612x904x3) verified readable |
| Annotated snapshots saved | 2 images (1612x904x3) verified readable |
| Evidence clips saved | 2 MP4 clips (1612x904, 30 FPS, 301 frames each, 10.0 s duration) |
| Clip coverage (BAR-001) | Bounds [33.20s -> 43.20s]; actual pre=5.00s / post=5.00s (301 frames) |
| Clip coverage (EXC-002) | Bounds [36.20s -> 46.20s]; actual pre=5.00s / post=5.00s (301 frames) |
| Clip reopen verification | PASS: both clips reopened, readable first frames, dimensions and FPS verified |
| Metadata JSON files | 2 files verified valid JSON with required relative paths and timing |
| Audit log | `evidence/cam_good_test/milestone5_test/audit.jsonl` (4 valid JSON Lines: 2 `EVENT_CREATED`, 2 `EVIDENCE_SAVED`) |
| Unit tests | PASS: 89/89 tests (including 22 dedicated evidence tests and all prior regressions) |
| Dependency check | PASS: `pip check` found no broken requirements |
| Environment check | PASS, exit 0: PyTorch CUDA on GeForce GTX 1650 |

| File | Purpose, input and output |
| --- | --- |
| `src/events/models.py` | Common `SafetyEvent` schema, ID generator, and adapters for BAR/temporal events |
| `src/events/audit.py` | Append-only `AuditLogger` writing JSON Lines audit records |
| `src/events/evidence.py` | `EvidenceWriter` capturing raw/annotated snapshots, clamping video clips, and saving metadata |
| `src/events/__init__.py` | Package symbols export |
| `scripts/run_evidence_pipeline.py` | Recorded video evidence runner combining detector, tracker, rules, and evidence capture |
| `tests/test_evidence.py` | 22 comprehensive unit tests covering all 25 Milestone 5 requirements |

**MILESTONE 5 STATUS: COMPLETE.**

### Milestone 5 visual evidence review results

Manual prototype visual review completed and accepted by the user:

- **EVIDENCE VISUAL ACCEPTANCE: PASS**
- **BAR-001 evidence review: PASS**
  - Raw snapshot: clean original CCTV imagery (1612x904)
  - Annotated snapshot: corresponds to BAR-001 event; shows Track 9, configured restricted-zone polygon, and BAR-001 ENTRY banner
  - Evidence clip: ~10.0 seconds total (bounds 33.20s -> 43.20s; frame 996 to 1296); visually provides correct ~5s pre-event, event (38.200s, frame 1146, Track 9 in `restricted_zone_1`), and ~5s post-event context
- **EXC-002 evidence review: PASS**
  - Raw snapshot: clean original CCTV imagery (1612x904)
  - Annotated snapshot: corresponds to EXC-002 event; displays `EXC-002 LONE WORKER` banner, Track 9, and configured zone polygon
  - Evidence clip: ~10.0 seconds total (bounds 36.20s -> 46.20s; frame 1086 to 1386); visually provides correct ~5s pre-event context (approx. 36.2s), event time (41.200s, frame 1236, Track 9 in `restricted_zone_1`), and ~5s post-event context (approx. 46.2s)
- **ERG-006 evidence review:**
  - 0 events recorded and 0 evidence packages created.
  - Correct and expected because `no_movement_seconds = 600`, while the test video is ~98.8s long. No fabricated evidence. Threshold remains 600s.
- **Audit log review:**
  - `evidence/cam_good_test/milestone5_test/audit.jsonl` verified with 4 valid JSON Lines: 2 `EVENT_CREATED` and 2 `EVIDENCE_SAVED`.

### Milestone 5 limitations (preserved)

- Evidence capture and audit trail functionality are prototype-level software mechanisms.
- `audit.jsonl` is an append-only JSON Lines file for local traceability; it is not tamper-proof, immutable, forensically certified, or regulatory-compliant.
- Severity remains prototype standard `"unclassified"` (automated severity classification `EVD-003` is deferred).
- Status remains `"new"` (acknowledgement workflow `EVD-011` is not implemented).
- Temporary Track IDs are strictly camera/session-local; no employee identity or cross-camera identity is inferred.
- Visual review is manual prototype verification on a single test video, not formal safety certification or statistical accuracy validation.

## Milestone 6: one live RTSP camera (PLT-001)

Milestone 6 adds only one fixed-view live source, `live_cam_1`. Non-secret
metadata is stored in `config/cameras.yaml`; the RTSP value is resolved at
runtime exclusively through `SAFETYSHIELD_RTSP_LIVE_CAM_1`. The ignored `.env`
was confirmed by `git check-ignore -v .env`. No URL is stored in YAML, source,
documentation, metrics, CLI arguments, or Git.

`src/camera/rtsp_reader.py` owns OpenCV connection/decode/reconnect state. One
reader thread publishes into a single protected latest-frame slot. Replacing an
unconsumed frame increments `frames_overwritten_or_dropped` once; a completed
inference increments `frames_processed`. Any remaining delivered/latest frame
is classified as dropped during shutdown, so the final run satisfies:
`frames_received = frames_processed + frames_overwritten_or_dropped`.
There is no queue and no multiprocessing.

The reader uses monotonic elapsed time, a stop event, bounded capture timeouts,
bounded thread join, safe capture release, five initial connection attempts,
and reconnect backoff of 1, 2, 4, 8, then at most 10 seconds. A valid decoded
frame resets consecutive failures and backoff. Installed OpenCV 5.0.0 exposes
`CAP_FFMPEG`, `CAP_PROP_OPEN_TIMEOUT_MSEC`, and
`CAP_PROP_READ_TIMEOUT_MSEC`; the prototype supplies 5,000 ms open/read values
through the FFmpeg capture constructor, with a constructor fallback for builds
that reject parameters.

The dedicated `scripts/run_rtsp_pipeline.py` loads the environment internally,
runs the locked `yolo26n.pt`, `imgsz=960`, confidence 0.20 person-only CUDA 0
detector and unchanged ByteTrack defaults, and uses monotonic live-session
timestamps. Actual decoded `frame.shape` is authoritative and remains native;
resolution changes are detected and keep fixed geometry disabled. Optional
review output and one clean raw reference frame are written under ignored
`outputs/`.

No `live_cam_1` zone exists. The 1612x904 `restricted_zone_1` polygon remains
owned by `cam_good_test` and was not reused. Therefore BAR-001, IDT-004,
EXC-002, and all temporal-rule execution were disabled for the live run while
person detection and camera/session-local ByteTrack continued. No zone event or
event evidence was fabricated. Milestone 5 can reopen finite recorded MP4s for
evidence; it does not provide pre-event context for an endless live source.
Large live raw-frame buffers and live pre/post evidence remain deferred.

### Actual 120-second live acceptance result

The command documented in README ran once with display disabled, annotated
review output enabled, and third-party stderr suppressed to avoid accidental
credential disclosure. Session ID was `live_acf176649da2`.

| Check | Actual observed result |
| --- | --- |
| Secret configured | YES; source name only: `SAFETYSHIELD_RTSP_LIVE_CAM_1` |
| Connection | SUCCESS; 1 attempt, 1 successful connection, 0 reconnects |
| Time to first decoded frame | 1.957 seconds |
| Native decoded resolution | 2560x1440 (validated from frame shape) |
| Reported source FPS | 20.0 |
| Acceptance duration | 120.029 seconds |
| Frames received | 2,434 (~20.28 received/s over acceptance time) |
| Frames processed | 1,229 |
| Frames dropped/overwritten | 1,205 (intentional latest-frame replacement) |
| Failed reads | 0 |
| Processing throughput | 10.239 FPS |
| Peak PyTorch GPU memory | 98.711 MiB allocated |
| Temporary track IDs observed | 2, scoped to this camera/session only |
| Zone status | Disabled / not configured |
| Zone and temporal rules | Disabled |
| Live evidence | Deferred; no live pre/post buffer or event evidence |
| Shutdown | Clean; capture released and reader thread joined |
| Raw reference | `outputs/live_cam_1_reference.jpg`; readable 2560x1440 |
| Annotated review | `outputs/live_cam_1_rtsp_test.mp4`; readable 1,229 frames, 2560x1440, 20 FPS (61.45s playback timeline) |
| Unit/regression tests | PASS: 110/110, including 21 offline RTSP tests |
| Dependency check | PASS: no broken requirements |
| Environment check | PASS, exit 0; CUDA calculation on GTX 1650 |

Timing semantics note:
Authoritative live session elapsed time is measured strictly via `time.monotonic()` (120.029 seconds).
The existing previously generated review recording `outputs/live_cam_1_rtsp_test.mp4` contains 1,229 processed frames encoded at the source camera rate of 20.0 FPS, resulting in an encoded playback duration of 61.45 seconds (~2x time-compressed); this existing file remains time-compressed and is not rewritten.
Any timestamps derived from `output_video_frame / output_fps` represent only the review-video timeline, not live monotonic session elapsed time.
To address this for future recordings, `scripts/run_rtsp_pipeline.py` adds a configurable `--output-fps` parameter that defaults to a cap of 10 FPS (`min(source_fps, 10.0)`). This is a prototype playback choice that approximately matches the current GTX 1650 processing rate (~10.24 FPS), rather than an exact live-time reconstruction. Future recordings using this default will have playback timing closer to real session duration, but timing is not guaranteed exact. Latest-frame dropping remains unchanged and no artificial frames are duplicated.

The live rate includes transport, decode, inference, tracking, and deliberate
stale-frame replacement and is not directly comparable to recorded-video FPS.
No synchronized camera timestamp was available, so no numeric end-to-end
network latency is claimed. The latest-frame design prevented a growing frame
queue. The raw reference shows the native fixed view; sampled review frames
show readable LIVE/performance/zone-disabled overlays. Two temporary tracks
were produced, but correctness is not asserted without the user's visual
review or labelled ground truth.

### Milestone 6 visual review result

The user completed manual visual review of the native reference frame (`outputs/live_cam_1_reference.jpg`), sample contact sheet (`outputs/live_cam_1_review_samples.jpg`), and individual full-resolution review frames extracted under `outputs/live_cam_1_track_review/`.

1. **Camera visual acceptance (PASS)**:
   - Stable fixed-view CCTV perspective with native 2560x1440 imagery and no apparent camera motion across reviewed samples.
   - Clean, readable live overlay showing camera ID, processing FPS (~10 FPS), and zone status.
   - Zone status correctly displayed as disabled / not configured. The old 1612x904 `restricted_zone_1` polygon belonging to `cam_good_test` was correctly not reused.

2. **Track 1 visual acceptance (PASS)**:
   - Evaluated on frames `track1_t26.0s_frame_0520.jpg` and `track1_t26.2s_frame_0524.jpg` (note: 26.0s / 26.2s are review-video playback timestamps, not live monotonic session elapsed time).
   - Shows a real distant worker with valid person bounding box and temporary Track 1 label visible and consistently assigned across nearby frames.
   - Reasonable bounding-box alignment for distant worker scale. Prototype validation only; no formal distant-person accuracy benchmark is claimed.

3. **Track 2 visual acceptance (PASS)**:
   - Evaluated on frames `track2_t31.6s_frame_0632.jpg`, `track2_t32.2s_frame_0644.jpg`, `track2_t33.0s_frame_0660.jpg`, and `track2_t33.8s_frame_0675.jpg` (review-video playback timestamps).
   - Shows a clearly visible worker with consistent temporary Track 2 label followed across the slope and foreground across multiple frames.
   - Good bounding-box continuity and alignment during posture changes and movement. Prototype validation only; no formal MOTA/HOTA/IDF1 metrics are claimed.

4. **Overall visual conclusion**:
   - Visual acceptance validates prototype stability, person detection, temporary camera/session-local tracking continuity, and safe zone-disabled behavior on this live RTSP feed.
   - Does not assert formal labelled detection accuracy, MOT tracking benchmarks, cross-camera identity, or safety/production certification.

Pre-existing Antigravity/Pyrefly diagnostics remain in the Milestone 0 environment-check script (`scripts/check_environment.py`); runtime validation passes (exit 0) and they are not a Milestone 6 blocker.

| File | Purpose, input and output |
| --- | --- |
| `src/camera/rtsp_reader.py` | Secret-safe camera config, one-slot reader, timeout/reconnect logic, metrics, resolution detection, and shutdown |
| `scripts/run_rtsp_pipeline.py` | Finite one-camera detector/tracker runner, overlay, raw reference, configurable `--output-fps`, and annotated review output |
| `tests/test_rtsp_reader.py` | Offline fake-capture tests for config security, connection, frame accounting, reconnect, resolution, shutdown, zones, reference saving, duration, and output FPS |
| `config/cameras.yaml` | Non-secret `live_cam_1` metadata and approved environment-variable name only |

**MILESTONE 6 AUTOMATED STATUS: PASS.**

**LIVE RTSP VISUAL ACCEPTANCE: PASS.**

**MILESTONE 6 STATUS: COMPLETE.**

## Milestone 7 basic dashboard and module controls

Automated acceptance run date: 2026-09-11. The user explicitly authorized
Streamlit installation and implementation of Milestone 7 basic dashboard and
module controls. No upgrade of Python, PyTorch, torchvision, CUDA, Ultralytics,
or OpenCV was performed. Milestone 8 was not started.

### Scope and architectural design

Milestone 7 provides an operator-facing prototype dashboard on localhost
`127.0.0.1:8501`. It is designed strictly for local development and review:

1. **Decoupled execution / zero inference on rerender**:
   - The dashboard does not run OpenCV RTSP readers, trigger YOLOv26 inference,
     or run PyTorch GPU operations on page load or widget interactions.
   - It reads configuration files (`config/cameras.yaml`, `config/zones.yaml`,
     `config/dashboard_controls.yaml`), runtime status telemetry JSON files
     under `outputs/runtime/`, and saved evidence packages under `evidence/`.
   - The dashboard is completely non-intrusive and cannot cause memory leaks or
     inference lockups.

2. **System status panel**:
   - Displays Python 3.14.5, platform details, virtual environment path,
     PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, and CUDA readiness on the
     NVIDIA GeForce GTX 1650 (4.0 GiB VRAM).
   - Verifies project repository structure.

3. **Configured cameras overview**:
   - Reads `config/cameras.yaml` to display registered cameras (`cam_good_test`,
     `live_cam_1`).
   - Secret-safe presence check: Evaluates whether `SAFETYSHIELD_RTSP_LIVE_CAM_1`
     is configured in the environment and displays a boolean indicator without
     ever logging or revealing the credential value.

4. **Live camera status & telemetry**:
   - `scripts/run_rtsp_pipeline.py` supports writing atomic runtime status to
     `outputs/runtime/<camera_id>_status.json`.
   - The dashboard monitors this telemetry with a 30-second freshness threshold.
     Status is classified as `running` (fresh), `stale` (no updates for >30s),
     `stopped` (cleanly terminated), or `offline` (no status file).
   - Metrics displayed: native decoded resolution, source FPS, processed FPS,
     frames received/processed/dropped, failed reads, reconnects, and temporary
     tracks.
   - If present, the clean native reference frame (`outputs/live_cam_1_reference.jpg`)
     is rendered.

5. **Module toggles & architectural dependency validation**:
   - Configured via `config/dashboard_controls.yaml` (schema 1.0).
   - Strictly enforces module hierarchy:
     - Person Detection is the root vision module.
     - Person Tracking depends on Person Detection.
     - BAR-001 (Restricted Zone Entry), IDT-004 (Zone Occupancy), and EXC-002
       (Buddy-Required Zone) depend on Person Detection, Person Tracking, and
       a valid camera-specific zone.
     - ERG-006 (Prolonged Low Movement) depends on Person Detection and
       Person Tracking; it does NOT require a zone.
   - Explicitly distinguishes "Global Enabled" toggle state from "Camera Available".
   - **Zone guard**: `live_cam_1` has no validated zone polygon (`cam_good_test`
     zone 1612x904 must not be reused for 2560x1440 live stream). Therefore, the
     dashboard marks zone-dependent modules (BAR-001, IDT-004, EXC-002) as
     camera-unavailable for `live_cam_1`, while ERG-006 remains camera-available
     when detection and tracking are enabled.

6. **Recent events & evidence explorer**:
   - Discovers structured event packages across `evidence/<camera_id>/<session_id>/`.
   - For any selected event, displays event ID, rule ID, timestamp, temporary
     Track ID, un-annotated raw snapshot, annotated snapshot with bounding boxes
     and banners, playable evidence video clip, and full metadata JSON.

7. **Prototype audit log viewer**:
   - Safely discovers `audit.jsonl` files across the evidence hierarchy
     (`evidence/<camera_id>/<session_id>/audit.jsonl` or session paths).
   - Deduplicates records, tolerates missing or malformed JSONL files without
     crashing, and preserves camera/session context.
   - Parses records into a readable tabular log showing UTC timestamp, action,
     camera ID, session ID, module ID, event ID, and details.
   - Clearly noted as an internal prototype development log (not immutable or
     forensically certified).

| Check | Actual observed result |
| --- | --- |
| Streamlit installation | `streamlit==1.63.0` installed in existing venv; no core packages altered |
| Dependency check | PASS: `pip check` found no broken requirements |
| Environment check | PASS, exit 0: torch 2.14.0+cu130, torchvision 0.29.0+cu130, CUDA on GTX 1650 |
| Unit tests | PASS: 139/139 tests passed, including 29 new tests in `tests/test_dashboard.py` |
| Local server launch | Tested on `http://127.0.0.1:8501`; returns HTTP 200 and `/stcore/health` returns 200 ok |
| AppTest automated run | PASS: `AppTest.from_file("scripts/run_dashboard.py").run()` executes cleanly |
| Secret safety | PASS: no secrets in code, logs, telemetry JSON, or UI |

### Milestone 7 visual review result

DASHBOARD VISUAL ACCEPTANCE: PENDING USER REVIEW.

| File | Purpose, input and output |
| --- | --- |
| `config/dashboard_controls.yaml` | Module toggle configuration schema and default enabled states |
| `src/dashboard/models.py` | Typed dataclasses for system status, camera info, telemetry, controls, events, and audit |
| `src/dashboard/data.py` | Secret-safe data loading, dependency logic, telemetry parser, evidence explorer, and audit loader |
| `scripts/run_dashboard.py` | Streamlit operator-facing dashboard script with layout, metrics, controls, and media |
| `tests/test_dashboard.py` | 29 offline unit tests verifying status, controls, dependencies, telemetry, evidence, and safety |
| `scripts/run_rtsp_pipeline.py` | Updated with `--runtime-status` atomic JSON status writer for live telemetry |

**MILESTONE 7 AUTOMATED STATUS: PASS.**

**DASHBOARD VISUAL ACCEPTANCE: PENDING USER REVIEW.**

**MILESTONE 7 STATUS: IMPLEMENTED.**

Next milestone: Milestone 8 (NOT STARTED / requires separate authorization).

Technical references consulted for the environment checks:
[PyTorch local installation and verification](https://pytorch.org/get-started/locally/)
and [Ultralytics installation](https://docs.ultralytics.com/quickstart/).
Use current official guidance when dependencies are changed; observed local
versions above describe this audit, not a recommended version combination.
