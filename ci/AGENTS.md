# Updating CI Toolchain Images

Read [README.md](README.md) before changing image configuration. Keep edits scoped
to the requested toolchain or release; do not change Jenkins or product coverage
as part of an image update.

## Configuration Ownership

- `toolchains.json`: CUDA requirements keyed by the POM's major.minor version.
  Update the target release entry only. Change `default` deliberately for future
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

## Contracts To Preserve

- Main publishes only `main-*`; `release/YY.MM` publishes only `YY.MM-*`.
  A release branch must match the POM version and have explicit toolchain pins.
  Main alone may use the configuration's default when its version entry is absent.
- Consumer lookup is different: both main and release GPU CI should try the POM's
  major.minor image first, then the matching `main-*` image. The CUDA, OS and
  architecture must match. Do not use the publishing prefix as the first-choice
  consumer tag. Consumer lookup is not implemented by the image-build workflows.
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
- Internal consumers use the same repository suffix, tags and digests through the
  GHCR remote cache, with their existing registry credentials. Do not add another
  publisher or a moving pre-release alias. Resolve source POM/toolchains, not job
  names; verify service-account cold-cache layer pulls and pin approved release/RC
  digests. An internal 404 may mean missing Deploy/Cache permission, not an absent
  upstream image; never silently mask that as normal image fallback.
  Compare upstream and cached digests after publication; matching recipe labels
  alone do not prove that a mutable tag's cached manifest is current.
  Exercise the dev-to-pre-release pipeline copy during adoption: copied code must
  resolve the actual source ref before allocating image-backed build agents, not
  retain a hardcoded `main-*` image or CUDA version from the dev job.

## Validation And Documentation

Run Bash syntax checks, ShellCheck, actionlint on the affected workflows and
`git diff --check`. Exercise `image-config.sh matrix` with the selected
`CI_TOOLCHAINS` map and both `IMAGE_TAG_PREFIX=main` and the POM's major.minor
prefix. Check main's missing-version default, release explicit-pin requirements,
branch/POM mismatch rejection and patch/snapshot tag reuse. A configuration change
must affect the actual generated build arguments, not only documentation.
Check both architectures, including rejection of mismatched platform/build args
and a smoke-inspected image with the wrong OS or architecture. For internal mirror
adoption, audit the actual dev/pre-release/release and legacy consumers first;
preserve verified digests and do not retire old image variants implicitly.

Update README examples and the PR description for changed tags, ownership or
fallback behavior. Keep the scenario table current. Distinguish local checks, PR
build/smoke validation, actual GHCR publication and future consumer validation;
do not claim that a successful image build proves consumer fallback works.
