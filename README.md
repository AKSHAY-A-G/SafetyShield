# SafetyShield

Construction/factory computer vision safety monitoring prototype

Current stage: Milestone 4 complete after manual prototype visual acceptance.
Milestone 5 has not started and requires separate authorization.

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
.\venv\Scripts\python.exe -B scripts\run_person_detection.py --input data\raw_videos\cam_good_test.mp4 --imgsz 960 --confidence 0.20
```

The input must be a readable local MP4. The command uses CUDA device 0 and the
selected configurable `imgsz=960`, confidence 0.20 settings. It writes
`outputs\cam_good_test_person_detected.mp4`, reports measured processing FPS and
PyTorch peak allocated GPU memory, then reopens the output to check its frame
size and readability. A missing input, invalid FPS, unavailable CUDA device or
unsupported output codec is reported as an error. Generated videos and model
weights are ignored by Git.

To reproduce the matched-frame comparison after generating the two experiment
videos, run:

```powershell
.\venv\Scripts\python.exe -B scripts\extract_detection_comparisons.py `
  --baseline outputs\person_detection_640_conf025.mp4 `
  --higher outputs\person_detection_960_conf020.mp4 `
  --output-dir outputs\comparison `
  --timestamps 10 30 50 70 80
```

This writes ignored, side-by-side PNGs from matching frame indices. They are for
manual comparison only and do not measure detection accuracy.

Manual review of the five comparisons accepted 960/0.20 as the Milestone 1
prototype baseline because it recovered additional valid medium/distant workers
without an obvious false-positive problem in those samples. This is not a
labelled accuracy benchmark; formal precision, recall and mAP evaluation remains
for Milestone 12. The settings are configurable rather than scientific constants.

Run Milestone 2 camera-local ByteTrack processing from the project root:

```powershell
.\venv\Scripts\python.exe -B scripts\run_person_tracking.py --input data\raw_videos\cam_good_test.mp4
```

This reuses the selected Milestone 1 detector, writes
`outputs\cam_good_test_tracked.mp4`, and extracts review frames under
`outputs\tracking_samples\`. Labels such as `Track 7` are temporary within one
camera/session; they are not employee IDs or cross-camera identities. The run
reports throughput, PyTorch peak allocated GPU memory, temporary IDs appearing
in output and maximum simultaneous tracks. Visual review is required to assess
ID stability, occlusion recovery and swaps.

The initial review found apparent fragmentation, so candidate B kept the
detector fixed and tested ByteTrack high/new thresholds of 0.20 with a 45-frame
buffer. Its separate full-video output is
`outputs\cam_good_test_tracked_candidate_b.mp4`. Generate matching A/B contact
sheets without rerunning detection or tracking:

```powershell
.\venv\Scripts\python.exe -B scripts\extract_tracking_comparisons.py `
  --baseline outputs\cam_good_test_tracked.mp4 `
  --candidate outputs\cam_good_test_tracked_candidate_b.mp4 `
  --output-dir outputs\tracking_tuning\comparison_sheets
```

The ignored sheets cover 5-15, 25-35 and 75-85 seconds at 0.5-second intervals.
Manual prototype review accepted candidate B without an obvious visual
regression in those samples. The selected configurable ByteTrack defaults are
high 0.20, low 0.10, new 0.20, buffer 45, match 0.80 and score fusion enabled.
A temporary Track ID count is not a unique-person count. This review is not
labelled ground truth and establishes no formal tracking metric.

Run the authorized Milestone 3 fixed-camera zone rules from the project root:

```powershell
.\venv\Scripts\python.exe -B scripts\run_zone_rules.py `
  --input data\raw_videos\cam_good_test.mp4 `
  --output outputs\cam_good_test_zones.mp4
```

The runner validates `config\zones.yaml` against the input resolution, reuses
the selected Milestone 1 detector and Milestone 2 ByteTrack settings, and uses
each temporary track's bottom-centre point for polygon membership. It overlays
the saved zone, current camera-local occupancy, tracks, bottom-centre markers,
inside state and BAR-001 entry banners. An entry requires confirmed OUTSIDE
followed by confirmed INSIDE; a track first appearing inside is counted after
confirmation but does not create an entry event. Generated videos and review
images remain ignored by Git. Manual review of the seven event-focused frames
accepted the outside, confirmation, single entry, inside, exit and occupancy
transitions for the prototype. The one-second banner display does not represent
additional events. This review is not a labelled accuracy evaluation.

Run the authorized Milestone 4 temporal rules from the project root:

```powershell
.\venv\Scripts\python.exe -B scripts\run_temporal_rules.py `
  --input data\raw_videos\cam_good_test.mp4 `
  --output outputs\cam_good_test_temporal_rules.mp4
```

This retains BAR-001 and IDT-004 while adding configured EXC-002 buddy-zone and
ERG-006 prolonged-low-movement state. `restricted_zone_1` is only a prototype
test zone; the system does not recognize confined spaces. EXC-002 uses a
configurable experimental 3-second confirmation and one event per continuous
single-occupancy episode. ERG-006 keeps the 600-second requirement and uses
bottom-centre displacement above `max(5 pixels, 10% of box height)` to reset
the timer. It is not medical or fatigue diagnosis. The runner writes ignored
review material under `outputs\temporal_rule_samples\`.

Manual review accepted the delayed EXC-002 confirmation, single event,
reset-pending debounce, continued episode without duplication and final clear
state. It also confirmed that low-movement time remained short or reset while
Track 9 moved. No ERG-006 event was expected or generated in the approximately
99-second clip under the unchanged 600-second requirement. This is prototype
visual acceptance, not formal safety accuracy evaluation.

Do not recreate `venv`. Package changes require explicit authorization.
