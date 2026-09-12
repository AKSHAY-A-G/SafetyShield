# SafetyShield - PPE Annotation Guide and Roboflow Workflow

This guide establishes the annotation standard and manual labeling workflow for the SafetyShield Personal Protective Equipment (PPE) detection model.

---

## 1. PPE Detection Architecture

In SafetyShield, PPE monitoring operates as a two-stage computer-vision pipeline:

1. **Stage 1 (General Person Detection & Tracking)**:
   A pretrained nano detector (`yolo26n.pt`, `imgsz=960`, `conf=0.20`, CUDA 0) detects workers on the original high-resolution CCTV frame and ByteTrack maintains temporal tracks.
2. **Stage 2 (PPE Detection on Person Crops)**:
   Each detected person is cropped from the **ORIGINAL native-resolution frame** (with 12% padding clamped to image boundaries) and fed to the custom PPE detector.
3. **Stage 3 (Temporal PPE State Machine)**:
   Observations are temporally filtered across frames to produce track states: `PRESENT`, `MISSING`, or `UNKNOWN`.

> [!IMPORTANT]
> **Do not run the PPE model over only a resized 640px full CCTV frame.**
> For medium and distant workers, small PPE details (such as helmet chin straps or vest reflective stripes) disappear when the entire frame is downscaled. The PPE detector must run on crops taken directly from the native high-resolution frame.

---

## 2. The Four Initial PPE Classes

SafetyShield Milestone 8 defines **exactly four** initial object detection classes:

| Class ID | Class Name | Description | Bounding Box Definition |
| :--- | :--- | :--- | :--- |
| **0** | `helmet` | Safety helmet / hard hat present | Tightly enclose the visible helmet itself. |
| **1** | `no_helmet` | Uncovered head with helmet confirmed absent | Tightly enclose the visible uncovered head region. |
| **2** | `vest` | High-visibility safety / reflective vest present | Tightly enclose the visible high-visibility vest garment. |
| **3** | `no_vest` | Torso visible with high-visibility vest confirmed absent | Tightly enclose the visible upper torso/clothing area. |

Do **NOT** add other classes (such as `person`, `worker`, `face`, `boots`, `gloves`, `goggles`, `harness`, or `mask`) during Milestone 8.

---

## 3. Critical Rule: UNKNOWN is NOT a Detector Class

SafetyShield's safety state machine evaluates three operational states:
- `PRESENT`
- `MISSING`
- `UNKNOWN`

However, **UNKNOWN is NOT an object detection class.** Never train `unknown_helmet` or `unknown_vest`.

`UNKNOWN` is a downstream safety decision derived automatically when:
- Crop resolution is too low / worker is too distant
- Motion blur or camera defocus obscures details
- Occlusion prevents clear inspection of head or torso
- Person is cropped at the edge of the frame
- Detector confidence is below threshold
- Temporal observations across consecutive frames are contradictory

If a worker's head or torso cannot be clearly and confidently judged by a human annotator, **DO NOT LABEL IT**. Leaving an ambiguous region unlabeled allows the downstream state machine to classify it as `UNKNOWN`.

---

## 4. Strict Annotation Policies

### A. Helmet Policy
- **`helmet`**: Annotate when a safety helmet, hard hat, or bump cap is clearly identifiable on the worker's head. The box must tightly enclose the helmet itself, not the entire head or body.
- **`no_helmet`**: Annotate **ONLY** when:
  1. The worker's head region is clearly and sufficiently visible.
  2. The uncovered head / hair / bare head is clearly recognizable.
  3. You can positively and confidently conclude that a helmet is absent.
- **Do NOT label `no_helmet` when**:
  - The worker is facing directly away and the head is indistinguishable.
  - The head is partially hidden by scaffolding, equipment, or machinery.
  - The head is near the image boundary and truncated.
  - The image is blurred, noisy, or low-contrast.
  - Distance makes it impossible to distinguish bare hair from a dark hard hat.

### B. Vest Policy
- **`vest`**: Annotate when a high-visibility fluorescent vest, jacket, or safety harness with reflective bands is clearly visible. The box should enclose the visible vest garment area.
- **`no_vest`**: Annotate **ONLY** when:
  1. The worker's upper body / torso is clearly visible.
  2. Regular non-reflective clothing (e.g., plain shirt, jacket, overalls) can be clearly inspected.
  3. You can positively and confidently conclude that high-visibility PPE is absent.
- **Do NOT label `no_vest` when**:
  - The torso is hidden behind railings, materials, or vehicles.
  - The worker is carrying a large object that blocks the chest.
  - Low lighting or heavy shadows prevent determining whether clothing is reflective.
  - The worker is too far away to distinguish fabric texture or hi-viz accents.

### C. Architectural Rule: Non-Detection is NOT "Missing"
The absence of a `helmet` or `vest` positive detection does **NOT** allow the system to infer `no_helmet` or `no_vest`. Missing PPE alerts require explicit negative evidence or validated multi-frame absence confirmation.

---

## 5. Leakage-Resistant Source-Group Splitting

Frames sampled seconds apart from the same camera clip are nearly identical. If adjacent frames are split randomly between training and validation, the model simply memorizes the background, yielding falsely optimistic validation metrics.

SafetyShield strictly enforces **Source-Group Splitting**:
1. Frames are grouped into contiguous source blocks: `{camera_id}_{clip_name}_b{block_idx}`.
2. Entire source groups are assigned to `train` (70%), `val` (15%), or `test` (15%).
3. All person crops extracted from a source group inherit the parent frame's split.
4. Crops from the same source group **never cross split boundaries**.

---

## 6. End-to-End Workflow

### Step 1: Extract Native-Resolution Source Frames
Run the sampling utility on your authorized local video:
```powershell
.\venv\Scripts\python.exe -B scripts\extract_ppe_frames.py `
  --input data\raw_videos\cam_good_test.mp4 `
  --camera-id cam_good_test `
  --interval-seconds 2.0
```
This writes uncompressed native frames into `data/dataset/ppe/source_frames/` and logs `source_frames_manifest.csv` and `split_manifest.csv`.

### Step 2: Generate Original-Resolution Person Crops
Generate padded person crops using the accepted person detector:
```powershell
.\venv\Scripts\python.exe -B scripts\create_ppe_person_crops.py `
  --frames-dir data\dataset\ppe\source_frames `
  --split-manifest data\dataset\ppe\manifests\split_manifest.csv `
  --output-dir data\dataset\ppe\crops
```
This saves crops into `data/dataset/ppe/crops/{train,val,test}/` with 12% relative padding and writes `crop_manifest.csv`.

### Step 3: First Annotation Review Gate (30 to 50 Crops)
> [!TIP]
> **Do not annotate 500 images at once.**
> Start by uploading **only 30 to 50 representative crops** into Roboflow. Review and validate this initial batch first to align annotation quality before scaling up.

1. Create a project in Roboflow:
   - Project Type: **Object Detection**
   - Classes: `helmet`, `no_helmet`, `vest`, `no_vest`
2. Upload the sample crops.
3. Annotate according to the strict visibility policies above.
4. Export the dataset:
   - Format: **YOLOv8** / **YOLOv11** (Ultralytics detection format)
   - Do NOT apply artificial augmentations in Roboflow (SafetyShield handles training data augmentation locally).
   - Download the zip and extract to `data/dataset/ppe/roboflow_export/`.

### Step 4: Validate Exported Annotations
Run the automated validator:
```powershell
.\venv\Scripts\python.exe -B scripts\validate_ppe_dataset.py `
  --dataset-dir data\dataset\ppe\roboflow_export `
  --manifest data\dataset\ppe\manifests\split_manifest.csv
```
The validator checks:
- All four classes match exactly.
- Coordinate bounding boxes are valid normalized numbers in `[0, 1]`.
- No empty or orphan files.
- Zero source-group leakage across splits.

### Step 5: Visually Inspect Labels
Generate visual QA overlay renders:
```powershell
.\venv\Scripts\python.exe -B scripts\render_ppe_annotations.py `
  --dataset-dir data\dataset\ppe\roboflow_export `
  --output-dir data\dataset\ppe\reports\review_samples `
  --max-samples 30
```
Inspect the output images under `data/dataset/ppe/reports/review_samples/`:
- Bright Green = `helmet`
- Red = `no_helmet`
- Amber/Cyan = `vest`
- Magenta = `no_vest`

Ensure boxes are tight and ambiguous cases remain unlabeled.

### Step 6: Pre-Training Gate & Smoke Test
Evaluate the pre-training gate:
```powershell
.\venv\Scripts\python.exe -B scripts\train_ppe.py --data config\ppe_dataset.yaml --gate-only
```
If a valid, non-empty, leakage-free dataset is verified, run a conservative smoke test (1 epoch):
```powershell
.\venv\Scripts\python.exe -B scripts\train_ppe.py `
  --data config\ppe_dataset.yaml `
  --epochs 1 `
  --batch 2 `
  --imgsz 640 `
  --device 0
```
Checkpoints will be saved under `models/ppe/` (never overwriting the base person model `yolo26n.pt`).

---

## 7. Privacy and Security Reminders
- CCTV images from job sites contain human subjects. Treat all dataset files as local development data.
- **Do NOT commit image files, crops, or dataset zips to Git.** (Guarded by `.gitignore`).
- **Do NOT commit Roboflow API keys, tokens, or credentials.**
- All Roboflow interactions must be performed manually through the web browser.
