# Facial Visual Profile HCI

A local Gradio interface for comparing ResNet18, ResNet50 and predicted-landmark fusion on a single uploaded image. This repository contains the HCI application and an optional, separate local language-model trial. Research training, benchmarks and the paper are maintained in [facial-visual-profile](https://github.com/DeweyYip/facial-visual-profile).

## Run locally

Use Python 3.10 in a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r hci_compare/requirements.txt
python hci_compare/check_language.py
python hci_compare/check_ui.py
python hci_compare/app.py
```

Restore your existing verified asset ZIPs into `weights/` at this repository root:

- `r18_seed1_hci_assets.zip`
- `r50_m3_seed1_hci_assets.zip` (the backend also accepts the historical `(1)` filename)

Weights are not supplied by cloning this repository. Existing asset bindings, attribute ordering and vendor hashes are retained and checked by the backend. There is no public model download or hosted demo asserted here. Keep private uploads and credentials outside Git.

The imported environment pins facenet-pytorch 2.5.3; this records the HCI source environment and is not a claim that it matches the research detector's 2.6.0 runtime. Do not substitute package versions or rebind hashes silently. A fresh full dependency installation and real-weight inference were not repeated during repository import.

## Scope and behavior

- The interface compares model scores and validation-frozen decisions, not calibrated probabilities or verified personal facts.
- The input pipeline preserves the shared letterbox and corruption implementation. MTCNN runs on the final input canvas.
- M3 is unavailable when landmarks are unavailable; that state must not be displayed as a negative attribute prediction.
- The current UI does not load or start Qwen. `llm_trial/` is a separate optional experiment; see its README and pinned manifests.
- No user study, clinical validity or demographic fairness evaluation is claimed.

## Files

- `hci_compare/`: app, backend, styles, display and wording contracts, synthetic fixtures and checks.
- `hci_compare/vendor/`: hash-verified inference source snapshot; retained independently from evolving research code.
- `hci_compare/acceptance_nov02/`: historical frozen acceptance fixtures and metadata, not certification of the latest UI.
- `llm_trial/`: optional GGUF setup and controlled trial tools; model binaries and runtime downloads are excluded.
- `docs/import_notes.md`: import changes and validation scope.

HCI development and research experiments have separate repositories. Importing the HCI source does not change any research model, threshold or reported experiment result.
