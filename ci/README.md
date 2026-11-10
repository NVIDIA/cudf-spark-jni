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
After the native builds, one manifest job per CUDA version combines that run's
verified digests into a tag containing both platforms. It does not rebuild images
or use emulation. Missing, duplicate, stale-source or mismatched-recipe records
fail publication for that CUDA version; a failed CUDA family does not prevent the
other complete pair from publishing. Caches remain architecture-specific.

Publisher runs and cleanup share one package-wide lock across main and release
branches. An active run finishes while pending runs queue; each publisher still
builds up to four images in parallel. Separate locks serialize main and its
matching release publisher per release line, CUDA family and platform.
The combined manifest has a separate release-line/CUDA-family lock to protect
its version aliases across main and release publishers.
Publication builds fresh, checks anonymous pulls and smoke tests, then promotes the
verified digest. Main checks for the release branch after the build, while holding
that lock; only a confirmed absent branch permits its version alias. API or
publication-output errors fail closed. A release publisher cannot be overwritten
by an already-running main publisher during branch-out. Initial GHCR package
visibility may need to be made public before publication succeeds.

Daily cleanup retires build-only versions and unreferenced untagged manifests
older than seven days, capped at 20 versions per run. Live main/release aliases
and their dependencies remain protected. Manual cleanup defaults to dry-run;
drain older publishers and inspect a dry-run before the first scheduled deletion.

## Tags And Compatibility

Images live under `ghcr.io/nvidia/cudf-spark-jni/jni-build`, with tags such as:

```text
26.10-cuda13.3.0-rockylinux8
26.12-cuda13.4.3-rockylinux8
main-cuda12.9.1-rockylinux8
main-cuda13.4.3-rockylinux8
```

Each tag contains `linux/amd64` and `linux/arm64`; Docker selects the host platform,
or callers can specify `--platform`. CUDA 12 and CUDA 13 remain distinct tags.

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
tags are publication candidates only. A combined tag is a manifest index over the
verified native images; consumers validate the selected platform's recipe label,
not a shared index-level fingerprint.

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

The default is versioned-only: select the `YY.MM-*` multi-platform tag. Missing
pins or a missing/incompatible versioned image must fail. Development callers may
explicitly use `--allow-main-fallback` to permit the checkout's `main` pins when
the version entry is absent and append
one recipe-checked `main-*` candidate. Release callers must not enable this flag.
Publish the matching combined manifests before enabling consumers of the updated
selector.
The selector does not infer development/release policy from a branch name, commit
containment or the `SNAPSHOT` suffix, since the same commit may be used in both.

Consumers choose their registry endpoint and authentication. Before using either
candidate, verify its Linux platform and recipe label against the returned values,
then use the immutable digest. Development callers may try the next candidate;
authentication/network failures must not be treated as an absent tag. The selector
itself does not pull, publish or silently build an image. Existing product-test
Jenkinsfiles and the local `build/run-in-docker` workflow are unchanged.

Maven dependencies and plugins prefer Google's Central mirror by default, including
plain local Maven and Docker builds, with Central retained as fallback. Existing
developer settings are not replaced. The POM declares Google first, followed by
`maven-central-fallback`, for both dependencies and plugins. Internal `ci/settings.xml`
redirects both IDs to the same authenticated Artifactory entry as Central. External CI uses
`ci/settings-public.xml` to mirror Central explicitly, including repository entries
introduced by dependency/plugin POMs, without a separate repository profile.
The POM's fallback repository is not mirrored by these public settings and remains
available if Google cannot supply an artifact.

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
The guarded GPU worker supplies its read-only job token in a temporary,
GitHub-only netrc mounted into the build container. CMake uses `OPTIONAL` netrc
authentication; the token is not passed in the container environment or cached.
Console output and uploaded diagnostics are filtered for download credentials.
If container cleanup or filtering cannot be confirmed, evidence/cache uploads
are withheld without replacing the workload's exit status. Authentication is an
outage mitigation, not a guarantee of successful dependency downloads.

Other callers keep their current authentication defaults. To use a CI-prepared
netrc for all three native configure steps, export `CMAKE_NETRC=OPTIONAL` and
`CMAKE_NETRC_FILE=/path/to/netrc` before invoking the existing build script or Maven.
The caller owns that file's permissions, lifetime and log redaction. No file is
created or deleted by `build/buildcpp.sh`; existing local and Blossom entry points
remain supported, without a coordinated pipeline rollout.

For PRs targeting main or `release/YY.MM`, approve the exact head with
`/ok to test <full-head-SHA>`.
The bot enforces that SHA for untrusted changes; for trusted contributors its
current policy copies the latest head. This workflow verifies the copied SHA
against the open PR, rather than interpreting the approval comment itself.
The copied push selects suites from `.github/gpu-ci-suites.json`; documentation and
publisher/cleanup/Sonar/CPU-workflow-only changes skip GPU jobs while still reporting
the gate. Native/Java sources, build inputs, toolchain recipes and GPU CI scripts
select the tests. The existing PR-title
`[skip ci]` marker also skips GPU dispatch while reporting the gate.
Keep existing required checks and Blossom/Sonar enabled during validation.
The bot must be enabled and the worker registered with Actions before its first
dispatch. A first-PR pilot can register it with a temporary, scoped hosted job;
remove that registration job before validation because the queue action requires
exactly one worker job. No extra registration trigger remains in this workflow.
Existing release branches also need this workflow, its scripts/catalogs and matching
toolchain pins before their copied PRs can use it; merging only to main does not
enable them. Do not retire their existing Blossom checks before that rollout.

Workers use public GHCR images verified against the checkout's recipe, or build and
smoke-test a missing/incompatible recipe locally without publishing. For the shared
multi-platform tags, GPU workers select the linux/amd64 child, verify its recipe
label and pin its immutable digest; the arm64 image and attestation descriptors are
not used by these workers. Registry/auth failures remain errors.
Tests reuse `premerge-build.sh` and `fuzz-test.sh` with public
Maven settings. Only queued workers may switch GPU pools, never failed builds/tests.
Compiler parallelism uses available CPUs and memory, including cgroup limits,
with 4 GiB per compiler. `PARALLEL_LEVEL` can lower this limit; there is no fixed
18-job ceiling on larger runners.
The parent resolves the PR's actual target, and the worker rechecks it after waiting
for a runner. Release targets require matching POM/pins and versioned images; only
main targets may use a recipe-compatible main image fallback.
The GPU worker has a four-hour total budget, allowing fifteen minutes per GPU pool
within the hosted coordinator's six-hour limit. On workload entry, elapsed setup time
is deducted from a 210-minute budget to reserve time for evidence and optional cache
uploads. This does not impose a single deadline across all setup steps; a setup hang
can still exhaust the job budget. Jenkins retains its six-hour test budget.

`GPU cache preparation` is a separate, non-required post-merge workflow for main
and migrated `release/YY.MM` branches. CPU workers run
`premerge-build.sh --prepare-cache`: the same native targets/options and Java
test compilation, stopping at Maven `test-compile`, without tests or fuzzing.
Its failure does not change the PR's GPU gate or the result of product tests.
Documentation-only pushes skip warming; manual runs and uncertain/truncated push
diffs prepare all enabled suites. Release pushes dispatch the main workflow with
the pinned release source so caches are saved in main's scope, visible to copied
PR branches. The workflow and compatible helpers must be merged on main and
backported to supported releases before their automatic warming is available.

PRs restore their latest matching cache first, then main's trusted baseline; they
cannot read sibling PR caches or write main's cache. Maven can reuse dependencies
across CUDA families. Native caches use the checkout's release line, architecture,
CUDA family and toolchain recipe, not the branch name: main and a release with the
same version/recipe can share a baseline. Ccache still checks source/compiler/flags;
RAPIDS architecture targets keep entries reusable across GPU pools. Each run saves
a fresh snapshot, including reruns that enrich an earlier partial cache. No cached
test result replaces validation. Restore/upload errors remain best-effort.

Both GPU workers and build-only warmers also cache the RAPIDS CMake bootstrap and
Arrow's Thrift, RapidJSON and xsimd archives inside the persisted CPM directory.
`JNI_DOWNLOAD_CACHE` opts into this archive cache; other callers are unchanged.
URLs/pins and available upstream checksums identify entries; explicit Arrow URL
or mirror overrides win. Cache failures retain upstream downloads. Each build
still extracts, patches and configures dependencies in its own workspace.
This is not a complete dependency cache: pinned RAPIDS Git dependencies can still
download again. Generated build trees and test results are not cached.

Test output streams to the console. The parent summary links to each
worker; its `JNI result and diagnostics` step and job summary show the last phase,
failed tests, CUDA/GPU/driver, pinned image, restored cache keys, timings and ccache statistics,
plus a link to full reports. Failure excerpts are printed without requiring a download.
The artifact includes Surefire, fuzz (`monte.rmm.log`, `monte.state.log`), CMake and JVM
crash logs and is retained for two days. Collection is bounded and best-effort, before
cache uploads; it never changes the workload result and cannot be guaranteed after
cancellation or loss of the runner.
