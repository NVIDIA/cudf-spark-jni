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
selects its project-version entry, or the configuration's `default` entry when a new
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
platform and resolved build arguments, excluding build parallelism. The current
Dockerfile has no COPY/ADD inputs. Include any future file inputs in the fingerprint
before relying on reuse. Smoke checks verify Linux, the requested architecture and
the label, then run the inspected image ID, not a mutable tag. Temporary `build-...`
tags are publication candidates only. The architecture-suffixed tags are separate
images; there is no combined amd64/arm64 consumer tag.

Image CI always builds and smoke-tests from its checkout.

## Branching And Version Updates

- Add the release-line toolchain entry before branch-out. The release branch keeps
  its own pinned configuration and Dockerfile; later release updates publish only
  that release's versioned tags. Once the branch exists, main stops refreshing
  those aliases, even if main's POM has not advanced yet.
- Main may use `default` for a new project version without an explicit entry.
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
Unrelated changes skip the image jobs, not the gate.
These hosted PR jobs have read-only permissions and never publish images. They use
`ubuntu-latest` for amd64 and `ubuntu-26.04-arm` for arm64, both native 4-CPU runners;
GitHub currently has no `ubuntu-latest-arm` label. Publication uses the same native
runner labels. Existing Jenkins product-test coverage is unchanged.

Builds export separate layer-cache scopes for each full CUDA version and
architecture. The existing amd64 PR builds can also import earlier major-series
caches during migration. Cache export failure is non-fatal; build,
recipe-verification and smoke-test failures remain failures.
