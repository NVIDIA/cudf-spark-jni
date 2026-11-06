# CI Toolchain Images

JNI build toolchains for PR validation and nightly builds.

## Publication Ownership

**Main hands version-tag ownership to the release branch at branch-out.**
Additional tags reference the same verified digest; they do not rebuild the image:

| Source branch | Published tag prefix | Publication triggers |
| --- | --- | --- |
| `main`, before `release/YY.MM` exists | `main-` and current POM `YY.MM-` | Daily schedule, image-input push, manual dispatch |
| `main`, after `release/YY.MM` exists | `main-` only | Daily schedule, image-input push, manual dispatch |
| `release/YY.MM` | `YY.MM-` | Image-input push, manual dispatch |

Other branches cannot publish. A release branch must match the POM's major.minor
version and have an explicit entry in [`toolchains.json`](toolchains.json). Main
selects its project-version entry, or the configuration's `main` entry when a new
version has no entry yet. Each branch builds only its selected toolchains using its
own Dockerfile and the definitions in
[`image-builds.json`](../.github/image-builds.json), with up to four images at once
(two CUDA versions times two architectures).
Each selected CUDA version has separate native amd64 and arm64 builds on
GitHub-hosted runners, without consuming the shared CPU/GPU pools. Publication
compilation parallelism is set on the build worker, not the configuration job.

Publisher runs are serialized per branch. An active run finishes; only the latest
pending refresh for that branch is retained. A second, shared lock serializes main
and its matching release publisher per release line, CUDA family and platform.
Publication builds fresh, checks anonymous pulls and smoke tests, then promotes the
verified digest. Main checks for the release branch after the build, while holding
that lock; only a confirmed absent branch permits its version alias. API or
publication-output errors fail closed. A release publisher cannot be overwritten
by an already-running main publisher during branch-out. Initial GHCR package
visibility may need to be made public before publication succeeds.

## Tags And Compatibility

Images live under `ghcr.io/nvidia/cudf-spark-jni/jni-build`, with tags such as:

```text
26.10-cuda13.3.0-rockylinux8-amd64
26.12-cuda13.4.3-rockylinux8-amd64
main-cuda13.4.3-rockylinux8-amd64
main-cuda13.4.3-rockylinux8-arm64
```

Versioned tags use only major.minor: `26.12.0-SNAPSHOT` and `26.12.1` both use
`26.12-`. Patch versions and snapshot suffixes do not create new tags. The image
version label contains a recipe fingerprint of the Dockerfile,
platform, resolved build arguments and explicit `recipe-inputs` files. The shared
`container-image/metadata.py` helper computes it; the catalog excludes only build
parallelism with `recipe-exclude-args`. The current Dockerfile has no COPY/ADD inputs,
so each catalog entry has an empty `recipe-inputs` list. Before adding local COPY/ADD
inputs, list every source file in each affected entry, relative to the checkout root.
Also list any build-context configuration that changes the inputs, such as an added
`.dockerignore`. The helper deliberately does not parse Dockerfiles or discover
files automatically. For example, a recipe copying cuDF-Spark's
`integration_tests/requirements.txt` must list that path; changing its bytes then
changes the fingerprint. Update the CPU suite and image-push path filters for those
files too, so input-only changes trigger validation and publication.
Smoke checks verify Linux, the requested architecture and
the label, then run the inspected image ID, not a mutable tag. Temporary `build-...`
tags are publication candidates only. The architecture-suffixed tags are separate
images; there is no combined amd64/arm64 consumer tag.

Image CI always builds and smoke-tests from its checkout.

## Automatic Consumer Selection

Use the same checkout-derived metadata as the publisher instead of maintaining
image tags in each workflow:

```sh
# Point to a checkout of the public common repository, outside this source tree:
export CI_IMAGE_METADATA_HELPER=/path/to/common/container-image/metadata.py
python3 ci/image-config.py select --classifiers cuda12,cuda13
# Development consumers may opt in to compatible main fallback:
python3 ci/image-config.py select --classifiers cuda12 --allow-main-fallback
```

The selector needs Python 3.9+, the project checkout and the public common helper,
not Docker, Maven, registry credentials or a GPU. The GHA configuration jobs check
out common at `main`; other callers supply `CI_IMAGE_METADATA_HELPER` from their
own checkout of the same helper. No helper is downloaded or executed implicitly.
The `version` and `list` commands do not need the helper.
Selection returns each classifier's image repository, platform,
recipe fingerprint, candidate tags and build config. Resolve the requested ref to
a commit once and run the selector from that checkout; branch, tag and commit-SHA
inputs all use the same checked-out POM and recipe. Both architectures are selected
when `--classifiers` is omitted. Exact POM-version pins take priority.

The default is versioned-only: missing pins or a missing/incompatible `YY.MM-*`
image must fail. Development callers may explicitly use `--allow-main-fallback`
to permit the checkout's `main` pins when the version entry is absent and append a
recipe-checked `main-*` image candidate. Release callers must not enable this flag.
The selector does not infer development/release policy from a branch name, commit
containment or the `SNAPSHOT` suffix, since the same commit may be used in both.

Consumers choose their registry endpoint and authentication. Before using either
candidate, verify its Linux platform and recipe label against the returned values,
then use the immutable digest. Development callers may try the next candidate;
authentication/network failures must not be treated as an absent tag. The selector
itself does not pull, publish or silently build an image. Internal consumers resolve
these selections through the authenticated GHCR mirror rather than pulling directly
from GHCR. The local `build/run-in-docker` image workflow is unchanged.

Maven dependencies and plugins prefer Google's Central mirror by default, including
plain local Maven and Docker builds, with Central retained as fallback. Existing
developer settings are not replaced. Internal `ci/settings.xml` redirects the Google
repository to the same authenticated Artifactory entry as Central. External CI uses
`ci/settings-public.xml` to mirror Central explicitly, including repository entries
introduced by dependency/plugin POMs; that mode has no direct Central fallback.

## Branching And Version Updates

- Add the release-line toolchain entry before branch-out. The release branch keeps
  its own pinned configuration and Dockerfile; later release updates publish only
  that release's versioned tags. Once the branch exists, main stops refreshing
  those aliases, even if main's POM has not advanced yet.
- Main may use the `main` entry for a new project version without an explicit entry.
  This configuration fallback is separate from the `main-` image tag prefix,
  which is always owned by main. For example, main at `27.02.0-SNAPSHOT` publishes
  both `27.02-*` and `main-*` from one build until `release/27.02` exists.
- Changing CUDA creates a new tag suffix. Changing the Dockerfile/build arguments
  changes the fingerprint even when the tag remains the same. CMake stays in the
  branch's Dockerfile; there is no additional per-release CMake configuration.
- Until a new release's first successful publication, its versioned tag may be
  absent or still point to main's pre-cut image. Existing branches need these
  workflows before they can publish; adding them to main does not update other
  branches. Main does not refresh version tags for an existing release branch.
- Branch creation alone does not guarantee an image build: publication has
  image-input path filters, and a release cut may change only submodule metadata.
  Verify the first release publisher; manually dispatch it on `release/YY.MM` if
  needed before switching consumers. Keep existing images until it succeeds.

For future automated edits, follow the [agent update guide](AGENTS.md), including
the main/release ownership checks and validation requirements.

## PR Validation

`CPU CI gate` reports for PRs targeting main or `release/**`. Image-related PRs
validate that branch's selected toolchains and tag prefix without publishing.
The configuration job checks consumer selection against the publisher matrix,
including versioned-only selection and explicit main fallback, before image builds.
Unrelated changes skip the image jobs, not the gate.
These hosted PR jobs have read-only permissions and never publish images. They use
`ubuntu-latest` for amd64 and `ubuntu-26.04-arm` for arm64, both native 4-CPU runners;
GitHub currently has no `ubuntu-latest-arm` label. Publication uses the same native
runner labels. Existing Jenkins product-test coverage is unchanged.

Builds export separate layer-cache scopes for each full CUDA version and
architecture. The existing amd64 PR builds can also import earlier major-series
caches during migration. Cache export failure is non-fatal; build,
recipe-verification and smoke-test failures remain failures.

## GPU Premerge

`GPU CI` adds CUDA 12/13 amd64 native/Java tests and fuzzing alongside Blossom.
For PRs targeting main or `release/YY.MM`, approve the exact head with
`/ok to test <full-head-SHA>`.
The bot enforces that SHA for untrusted changes; for trusted contributors its
current policy copies the latest head. This workflow verifies the copied SHA
against the open PR, rather than interpreting the approval comment itself.
The copied push selects suites from `.github/gpu-ci-suites.json`; documentation-only
changes skip GPU jobs while still reporting the gate. Keep existing required checks
and Blossom/Sonar enabled during validation. The worker workflow and bot configuration
must be present on main before dispatch-based validation can run.
Existing release branches also need this workflow, its scripts/catalogs and matching
toolchain pins before their copied PRs can use it; merging only to main does not
enable them. Do not retire their existing Blossom checks before that rollout.

Workers use public GHCR images verified against the checkout's recipe, or build and
smoke-test a missing/incompatible recipe locally without publishing. Registry/auth
failures remain errors. Tests reuse `premerge-build.sh` and `fuzz-test.sh` with public
Maven settings. Only queued workers may switch GPU pools, never failed builds/tests.
The parent resolves the PR's actual target, and the worker rechecks it after waiting
for a runner. Release targets require matching POM/pins and versioned images; only
main targets may use a recipe-compatible main image fallback.
The GHA worker has a five-hour total budget; Jenkins retains its six-hour test budget.

Main pushes and manual main runs warm Maven, ccache and CPM caches using the same
worker. PRs can restore main's baseline and save branch-local updates. Native caches
are isolated by architecture and recipe; no compiled library or test result is used
as a substitute for validation. Optional cache failures do not turn build failures
into success. Test output streams to the console and reports are retained for two days.
