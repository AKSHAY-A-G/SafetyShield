# PPE training pool v1

`data/dataset/ppe/training_pool_v1/` is a copy-only **training-data-only** pool.
It combines the current SafetyShield site export with public supplemental PPE
data. Its `data.yaml` intentionally has no validation or test path, so it is
not an approved full-training dataset.

Public supplemental data is the **PPE Computer Vision Dataset** from Roboflow
Universe, licensed **CC BY 4.0**. The source URL, export identity/version, and
per-image attribution are retained in the pool manifest. The public data was
not captured by SafetyShield. Existing candidate-group IDs and review statuses
from `config/public_ppe_provenance_manifest.json` are retained per public image.

The current SafetyShield site images are also training candidates only. Future
validation and test evidence must be newly captured, independent SafetyShield
site clip/time groups; neither may reuse images or adjacent source episodes from
this pool. PPE full training and Milestone 9 remain unstarted.

To rebuild from the two unchanged source exports:

```powershell
.\venv\Scripts\python.exe -B scripts\build_ppe_training_pool.py
```

The builder refuses to overwrite an existing pool, detects destination-name
collisions before copying, validates every source label, and leaves the full
training gate blocked until independent evaluation data exists.
