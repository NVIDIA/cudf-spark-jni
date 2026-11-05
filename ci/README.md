# CI Toolchain Images

## Publication Ownership

**Only `main` publishes shared GHCR images.** The daily image workflow runs on the
main schedule, image-input pushes to main, and manual dispatches selecting main.
Release pushes do not publish; dispatches selecting another branch are skipped.
This prevents a release checkout from overwriting main's shared image tags.

The publisher expands every entry in [`toolchains.json`](toolchains.json), including
`default`, using the image definitions in
[`image-builds.json`](../.github/image-builds.json). It builds each image/tag once,
rejects conflicting recipes for the same tag, and runs at most two images at once.
Currently this covers CUDA 12.9.1, 13.3.0 and 13.4.3. All use main's Dockerfile.

Publisher runs are serialized. An active run finishes; the latest pending main
refresh replaces an older pending refresh. There is no cross-branch publishing
queue. Publication builds fresh, checks anonymous pulls and smoke tests, then
promotes the verified digest. Initial GHCR package visibility may need to be made
public before this succeeds.

## Tags And Compatibility

Images live under `ghcr.io/nvidia/cudf-spark-jni/jni-build`, with tags such as:

```text
cuda12.9.1-rockylinux8-amd64
cuda13.3.0-rockylinux8-amd64
cuda13.4.3-rockylinux8-amd64
```

Project versions, snapshot suffixes and branches are not part of the tag. The
image version label instead contains a recipe fingerprint of the Dockerfile,
platform and resolved build arguments, excluding build parallelism. The current
Dockerfile has no COPY/ADD inputs. Include any future file inputs in the fingerprint
before relying on reuse. Smoke checks verify the label and run the inspected image
ID, not a mutable tag. Temporary `build-...` tags are publication candidates only.

A shared tag does not guarantee compatibility with every checkout. The future JNI
GPU consumer must check the expected fingerprint and build locally when the image
is missing or incompatible. That consumer is not implemented by these workflows;
current image CI always builds and smoke-tests from its checkout.

## Branching And Version Updates

- Add or update release-line toolchains in main's `toolchains.json` before branch-out.
  Keep supported release entries in main so their images continue to be published.
- Main may use `default` for a new project version without an explicit entry.
  Release validation requires an explicit entry matching the POM release line.
- Changing only the project version does not change image tags or fingerprints.
  Changing CUDA creates a new toolchain tag; changing the Dockerfile/build arguments
  changes the fingerprint even when the tag remains the same.
- A release-only recipe change is validated locally and cannot overwrite a shared
  tag. If it must be prebuilt, first make the recipe compatible in main or design a
  separately owned image; do not enable release publishing for these shared tags.

## PR Validation

`CPU CI gate` reports for PRs targeting main or `release/**`. Image-related main PRs
validate the entire deduplicated publication matrix; release PRs validate only
their selected toolchains. Unrelated changes skip the image jobs, not the gate.
These hosted PR jobs have read-only permissions and never publish images. Existing
Jenkins product-test coverage is unchanged.

Builds export separate layer-cache scopes for each full CUDA version. PR builds can
also import earlier major-series caches during migration. Cache export failure is
non-fatal; build, recipe-verification and smoke-test failures remain failures.
