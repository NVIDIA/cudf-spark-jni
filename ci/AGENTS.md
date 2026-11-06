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

- Main always publishes `main-*`. Until `release/YY.MM` exists, it also aliases
  that POM version's `YY.MM-*` tags to the same verified digest, without rebuilding.
  Once the branch exists, only `release/YY.MM` may refresh `YY.MM-*`.
  A release branch must match the POM version and have explicit toolchain pins.
  Main alone may use the configuration's default when its version entry is absent.
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

## Validation And Documentation

Run Bash syntax checks, ShellCheck, actionlint on the affected workflows and
`git diff --check`. Exercise `image-config.sh matrix` with the selected
`CI_TOOLCHAINS` map and both `IMAGE_TAG_PREFIX=main` and the POM's major.minor
prefix. Check main's missing-version default, release explicit-pin requirements,
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
