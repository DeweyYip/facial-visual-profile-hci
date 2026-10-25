# Source import notes

Source: the author's `files.zip` and `file2 2.zip`, imported on 2026-10-26 (Asia/Shanghai).

- Restored copied filenames by removing the Finder suffix `拷貝`; placed them under `hci_compare/` and `llm_trial/`.
- Excluded macOS metadata, Python bytecode and historical acceptance run outputs. Retained synthetic development/acceptance fixtures, freeze metadata, source bindings and vendor hashes.
- Did not import weights, downloaded LLM runtimes, backups, private photos or a virtual environment.
- Preserved imported Python, CSS and JSON bytes. Existing per-component READMEs are historical development notes; root README gives clean-checkout instructions.

## Validation

Python syntax and JSON parsing passed. Every entry in `hci_compare/vendor_hashes.json` matched the imported vendor bytes. Credential-pattern scanning found no matching private keys or common token patterns (not a guarantee of comprehensive secret detection).

Passed `check_language.py` (60 synthetic fixtures), `check_local_wording.py` (9 tests), `check_ui.py` (5 tests), and `llm_trial/check_trial.py` (controlled runtime stubs). These are source/contract tests, not real-image accuracy, real Qwen inference, a user study or a fresh full application smoke test. The original acceptance freeze predates the latest UI and must not be represented as its certification.

Real inference requires the author's separately stored verified asset ZIPs. Missing weights are expected in a fresh clone. Imported dependency pins and source hashes have not been upgraded or rewritten.
