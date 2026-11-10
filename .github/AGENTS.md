# GitHub Workflows

Keep GHA-only helpers and settings under `.github/scripts/`; `ci/` is for shared
local, GHA and Blossom scripts. Read [GPU CI](scripts/gpu-ci/README.md) for this
workflow's expected behavior and `ci/README.md` for shared image contracts.

Keep suite catalogs, job IDs and result gates aligned. GPU CI is all-except-listed:
exclude documentation and unrelated workflow files, not whole `.github` folders.
Preserve bot approval, copied-head/target checks and existing Blossom/Sonar coverage.
Do not retry registry or workload failures, or weaken release image pinning.

Keep cache warming build-only and independent of the GPU gate. Trusted main/release
sources may warm the shared baseline; PR caches must stay isolated from it.
Maintain existing credential-redaction checks when moving helpers. Run workflow
lint and applicable script checks; do not claim unperformed live validation.
