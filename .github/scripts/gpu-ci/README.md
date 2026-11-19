# GPU CI

`GPU CI` runs the CUDA 12/13 amd64 native/Java tests and fuzzing. For PRs targeting
main or `release/YY.MM`, an authorized NVIDIA user triggers it with
`/ok to test <full-head-SHA>`.
Check the copied revision: trusted contributors' commands can copy the latest head
rather than the supplied SHA.

If a run fails for a non-code reason and the PR commit is unchanged, someone with
repository write access can open the parent `GPU CI` run in GitHub Actions and
select **Re-run failed jobs**. A new commit needs another `/ok to test` command
with its new full head SHA.

The suite catalog selects JNI source, shared build inputs and GPU CI configuration,
excluding Markdown documentation. `paths` accepts exact filenames or directory
prefixes ending in `/`; add relevant workflows explicitly. Sonar and unrelated
workflows do not select GPU tests. Unaffected changes still report the gate;
a PR-title `[skip ci]` marker also skips GPU work. [Sonar checks](../sonar-check/README.md)
run separately. Release branches need the workflows, helpers and toolchain pins
backported before their PRs can use GPU CI.

Workers reuse a verified, branch-compatible GHCR image or build a missing recipe
locally without publishing it. Registry failures remain errors. Runner fallback
applies only while queued, not after a build or test fails. Compilation parallelism
adapts to available CPU and memory. Scheduling details live in the workflow files.

## Build Caches

GPU tests reuse Maven dependencies, ccache and supported CPM sources. Generated
build trees and test results are not cached; some dependencies still download
again. Each ccache is limited to 3 GB. Cache restore/upload failures do not replace
the build or test result.

`GPU cache preparation` builds without running tests after merges to main or enabled
`release/YY.MM` branches, and can be started manually. There is no daily cron.
It warms CUDA 12/13 on amd64, with one active warmup and one pending run per source
branch. New merges replace the pending run; it selects the latest source when it
starts. An allocated warmup can finish its pinned source. An already-prepared
source, or only excluded changes since that successful build, skips compilation
while both matching cache archives remain available. Missing or incomplete caches
are rebuilt. Warming never gates PR tests.

Release warmups build the release's exact source and toolchain, not main's source.
They save in main's cache scope so copied PR branches can access them. Native
caches are separated by release line, CUDA family, architecture and image recipe.
PRs prefer their own matching cache, then the shared baseline; PRs cannot update
that baseline. Backport the workflow and helpers to enable release warmups.

## Downloads And Diagnostics

GPU tests, post-merge cache preparation, and the hosted Sonar compile use Google's
Maven Central mirror by default. To enable direct Central for a script invocation,
set `MVN_SETTINGS_FILE=settings-central-fallback.xml` in its environment.

GHA uses a temporary, read-only GitHub credential for dependency downloads;
credentials are excluded from caches and filtered from published diagnostics.
The shared native build forwards explicitly supplied `CMAKE_NETRC` settings to all
three CMake configurations, preserving defaults when callers do not opt in.

Follow the parent summary's worker links for console output and
`JNI result and diagnostics`: failed tests, last phase, GPU/toolchain identity,
cache statistics and timings. Full Java/native/fuzz and crash logs are linked as
artifacts, retained for two days. Diagnostics are best-effort, never change test
results, and may be unavailable after cancellation, runner loss or unsafe cleanup.
