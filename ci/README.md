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

Versioned tags use only major.minor: `27.02.0-SNAPSHOT` and `27.02.1` both use
`27.02-`. Patch versions and snapshot suffixes do not create new tags. The image
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
