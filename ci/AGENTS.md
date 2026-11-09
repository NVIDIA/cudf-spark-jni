# Updating CI Toolchain Images

Read [README.md](README.md) before changing image configuration. Keep edits scoped
to the requested toolchain or release; do not change Jenkins or product coverage
as part of an image update.

For GPU CI changes, keep `.github/gpu-ci-suites.json`, its workflow job IDs and
the result gate aligned. Preserve explicit bot approval for copied PRs and the trusted
main-scoped cache baseline from trusted post-merge main/release sources. Keep
existing Blossom/Sonar coverage until explicitly retired; registry failures and
real workload failures must not cause GPU retries.
Recheck the copied head and actual PR target; only main targets allow main-image
fallback. Do not claim the bot enforces a supplied SHA for trusted contributors.
Keep cache preparation build-only and independent of the required GPU gate.
Preserve PR source guards; only the main cache workflow may warm trusted release
sources into main's cache scope. Cache keys use the built source, not the workflow
SHA; never promote a PR's cache into the trusted baseline.

## Configuration Ownership

- `toolchains.json`: CUDA requirements keyed by the POM's major.minor version.
  Update the target release entry only. Change `main` deliberately for future
  main versions; do not overwrite it with an older release's requirements.
  Preserve other release entries so release-to-main merges retain their pins.
- `Dockerfile`: branch-local image recipe, including CMake and compiler defaults.
  Do not duplicate these defaults in `toolchains.json` without a requested design
  change. Dockerfile/build-argument changes must still affect the recipe fingerprint.
- `../.github/image-builds.json`: image definitions, platform, arguments and cache
  scopes. CUDA is resolved from `toolchains.json`, not hardcoded again here.
  Keep amd64 and arm64 variants for every selected CUDA family. Pass both
  `TARGETPLATFORM` and the matching `CMAKE_ARCH` explicitly; the Dockerfile's
  defaults target amd64. Keep cache scopes architecture-specific.
- Consumer tags omit architecture and resolve through a multi-platform index.
  Keep native build tags, classifiers and recipe fingerprints architecture-specific.
  Combine verified immutable digests from the same source revision only after both
  platforms succeed; parallel workers must never overwrite the shared tag.
- Every local Dockerfile COPY/ADD source must be in each affected image's explicit
  `recipe-inputs` list, relative to the checkout root. Keep it empty only while
  there are no copied inputs. For example, cuDF-Spark's copied
  `integration_tests/requirements.txt` must affect the fingerprint; apply the same
  rule if JNI adds a requirements file, installer, patch or other copied source.
  List build-context controls such as `.dockerignore` when added. Do not implement
  a Dockerfile parser or silently hash an arbitrary directory. Update the CPU-suite
  and publishing workflow path filters so these input changes select image CI.
  Verify that changing only a declared file changes the fingerprint.
- `image-config.py`: project branch/version, argument mapping and tag policy for
  publishers and consumers. Fingerprinting belongs to public common's
  `container-image/metadata.py`, supplied via `CI_IMAGE_METADATA_HELPER`.
  Keep `image-config.sh` as its compatible CLI entry point; do not
  duplicate this policy in workflow YAML or internal CI. Registry/authentication
  and digest lookup belong to the caller.
- Consumer selection is derived from the exact checkout and versioned-only by
  default, including detached commits. Only development callers may opt in to
  main fallback; do not infer this policy from branch containment or SNAPSHOT.
  Publisher branch ownership is separate from consumer selection policy.

## Contracts To Preserve

- Main always publishes `main-*`. Until `release/YY.MM` exists, it also aliases
  that POM version's `YY.MM-*` tags to the same verified digest, without rebuilding.
  Once the branch exists, only `release/YY.MM` may refresh `YY.MM-*`.
  A release branch must match the POM version and have explicit toolchain pins.
  Main alone may use the configuration's `main` entry when its version entry is absent.
  Check release ownership after the build, under the shared release-line/CUDA-family/
  platform publisher lock. API uncertainty must not permit an alias. Use the shared
  action's immutable publication output, not a fresh read of a mutable main tag.
- Keep daily publication main-only, release publication push/manual-only, PR jobs
  read-only, and publishing jobs separate from PR execution. Preserve anonymous
  pull and smoke checks before promotion, branch concurrency and worker limits.
  A release cut may not match image-input path filters. Verify its first publisher
  and dispatch it manually if needed before switching consumers to versioned tags.
  Both workflows use native GitHub-hosted runners; do not add shared CPU/GPU pool
  usage without a demonstrated need. Set publisher compilation parallelism on the
  actual worker. Do not move unapproved fork PR code onto a shared runner pool.
- Cache failures must not hide or replace build/smoke failures. Keep metadata and
  fingerprint checks; never treat registry authorization/network errors as a missing tag.
- Combined manifests have their own release-line/CUDA-family lock and repeat the
  release-ownership check before aliasing. Keep the selector contract at one
  versioned tag plus one optional main fallback. One complete CUDA pair may publish
  even if another family failed. Extend the existing CPU metadata checks when changing
  this manifest/selector contract.

## Validation And Documentation

Run Bash syntax checks, ShellCheck, actionlint on the affected workflows and
`git diff --check`. Check Python syntax and `image-config.py select` from exact
checkouts with the shared helper configured, with versioned-only and explicit
development fallback policies. Confirm `version`/`list` still work without the
helper and metadata operations fail clearly when it is missing. Exercise
`image-config.sh matrix` with the selected
`CI_TOOLCHAINS` map and both `IMAGE_TAG_PREFIX=main` and the POM's major.minor
prefix. Check main's missing-version fallback, release explicit-pin requirements,
branch/POM mismatch rejection and patch/snapshot tag reuse. A configuration change
must affect the actual generated build arguments, not only documentation.
Exercise absent/existing release refs, API/schema failures, missing publication
outputs and ownership transfer during a build. Check that aliases use the verified
digest and perform no additional build, and preserve shared publisher serialization.
Check both architectures, including rejection of mismatched platform/build args
and a smoke-inspected image with the wrong OS or architecture.

Update README examples and the PR description for changed tags, ownership or
configuration behavior. Keep the scenario table current. Distinguish local checks,
PR build/smoke validation and actual GHCR publication.
