# CI Toolchain Images

## Publication Ownership

**Main and release branches own different tags.** No branch publishes another
branch's image tags:

| Source branch | Published tag prefix | Publication triggers |
| --- | --- | --- |
| `main` | `main-` | Daily schedule, image-input push, manual dispatch |
| `release/YY.MM` | `YY.MM-` | Image-input push, manual dispatch |

Other branches cannot publish. A release branch must match the POM's major.minor
version and have an explicit entry in [`toolchains.json`](toolchains.json). Main
selects its project-version entry, or the configuration's `default` entry when a new
version has no entry yet. Each branch builds only its selected toolchains using its
own Dockerfile and the definitions in
[`image-builds.json`](../.github/image-builds.json), with at most two images at once.
Each selected CUDA version has separate native amd64 and arm64 builds on
GitHub-hosted runners, without consuming the shared CPU/GPU pools. Publication
compilation parallelism is set on the build worker, not the configuration job.

Publisher runs are serialized per branch. An active run finishes; only the latest
pending refresh for that branch is retained. Main and release publishers may run
concurrently because their tags are separate. Publication builds fresh, checks
anonymous pulls and smoke tests, then promotes the verified digest. Initial GHCR
package visibility may need to be made public before this succeeds.

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

Publishing ownership is separate from image selection. **GPU CI on both main and
release branches should try its own POM major.minor tag first**, then the matching
`main-` tag if the first image is absent or its recipe is incompatible. For
example, main at `26.12.0-SNAPSHOT` tries `26.12-cuda13.4.3-rockylinux8-amd64` before
`main-cuda13.4.3-rockylinux8-amd64`; it does not go directly to the main image because
the source branch is main. Keep the same CUDA, OS and architecture suffix. This
covers branch-out and a main version bump before a versioned image is published.
Do not treat registry authentication or network errors as proof that a tag is absent.
Verify the expected recipe fingerprint before reuse; if neither candidate is
available and compatible, build locally. These workflows prepare the images, not
that consumer lookup: image CI always builds and smoke-tests from its checkout.

## Branching And Version Updates

- Add the release-line toolchain entry before branch-out. The release branch keeps
  its own pinned configuration and Dockerfile; later release updates publish only
  that release's versioned tags.
- Main may use `default` for a new project version without an explicit entry.
  This configuration fallback is separate from the `main-` image tag prefix,
  which is always owned by main, even when main selected an explicit version entry.
- Changing CUDA creates a new tag suffix. Changing the Dockerfile/build arguments
  changes the fingerprint even when the tag remains the same. CMake stays in the
  branch's Dockerfile; there is no additional per-release CMake configuration.
- Until a new release's first successful publication, its versioned tag may be
  absent. Existing branches need these workflows before they can publish; adding
  them to main does not update other branches. Main does not publish release images
  on their behalf.
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

## Internal Registry Cache

Use a GHCR remote cache in the internal registry rather than building or publishing
the same image again. The repository path after the registry prefix, tag and digest
stay the same. For example:

```text
ghcr.io/nvidia/cudf-spark-jni/jni-build:main-cuda13.4.3-rockylinux8-arm64
<internal-registry>/<virtual-repository>/nvidia/cudf-spark-jni/jni-build:main-cuda13.4.3-rockylinux8-arm64
```

The upstream GHCR package must allow anonymous pulls. Internal jobs retain their
existing registry authentication; no GitHub token is needed to pull a public image.
Verify a cold-cache full layer pull with the internal CI service account, not only
manifest access or an already cached pull with a developer's account. The service
account needs Read and Deploy/Cache permission on the underlying remote repository.
An Artifactory 404 can indicate missing cache permission rather than a missing
upstream tag; disambiguate it before consumer fallback. See
[JFrog's remote-cache permission guidance](https://jfrog.com/help/r/how-does-the-deploy-permission-relate-to-remote-repositories).
Remote caching is on demand; mutable-tag refresh
depends on the registry's metadata-cache policy. Prewarming must fetch the required
platform's layers, not only its manifest. After publication, compare upstream and
cached digests before adopting a new tag; a cached 404 or stale manifest is not
fixed merely by setting a client's pull policy to always.

Use the same consumer selection rules externally and internally:

| Source used by the job | First image | Fallback image |
| --- | --- | --- |
| Main/dev at `26.12.0-SNAPSHOT` | `26.12-cudaX-rockylinux8-ARCH` | `main-cudaX-rockylinux8-ARCH` |
| Pre-release at `26.10.2-SNAPSHOT` | `26.10-cudaX-rockylinux8-ARCH` | Matching `main-*` only if compatible |
| Release/RC at `26.10.x` | Verified `26.10-cudaX-rockylinux8-ARCH` digest | Only a separately verified compatible image |

Resolve `X`, `ARCH` and the recipe from the checked-out source, not the job name or
the current main configuration. A new main image must not substitute a different
CUDA version or compiler recipe for a release. RC/release jobs should pin and record
the approved digest so subsequent tag refreshes cannot silently change their build.
Do not introduce a moving `pre-release-*` alias or overwrite legacy shared tags.
These are adoption rules, not changes to existing Jenkins consumers in this PR.
Release-roll automation may copy a dev pipeline into its pre-release counterpart
unchanged. A future consumer must therefore resolve the actual source ref and its
metadata before selecting the build-agent image; hardcoding `main-*` in the dev
pipeline would also send the copied pre-release pipeline to the main image. Keep
source-branch routing and image-tag selection separate.

For an explicit retained mirror instead of an on-demand cache, copy the published
digest with all referenced content (for example,
`skopeo copy --all --preserve-digests`), then verify the destination digest,
architecture, recipe and smoke checks. Caching the final image does not mirror its
base image or build-time download dependencies.

This producer does not replace every internal image: older CUDA/GCC, CentOS,
customer/private variants, signing/deployment images and product integration-test
images keep their existing paths until separately validated. A CUDA version still
required by a consumer must remain available even when main advances its toolchains.
Existing release branches need to adopt the publisher before their versioned tags
exist; merging this PR into main does not publish their images automatically.
