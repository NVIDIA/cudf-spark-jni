# Sonar check

Sonar runs for approved PR commits. Each new commit needs approval; title or
target-branch changes require rerunning the approved workflow.

```mermaid
flowchart LR
  A[Approved copied head] --> B[Hosted Java compile + Maven cache]
  B --> C[Hosted source and class preflight]
  C --> D[Internal Sonar analysis]
  D --> E[Hosted findings report + Sonar check gate]
```

[sonar-paths.json](../../sonar-paths.json) selects the full PR diff, including
deleted/renamed paths, and controls scanner inclusions. It covers Java/Scala,
C/C++, scripts, XML/HTML, workflow YAML and enabled Text/secrets formats.
Documentation-only and Dockerfile-only changes skip Sonar. Vendored/generated
files, CUDA and generic JSON/YAML are excluded. A `[skip ci]` title or an
unaffected diff also skips Sonar.

## Results

| Result | Gate | PR comment |
| --- | --- | --- |
| Title CI-skip or unaffected diff | Skipped | None |
| Clean analysis | Success | None |
| Non-blocking findings | Success | Findings summary |
| High/Blocker issues or failed quality gate | Failure | Findings summary |
| Compilation, infrastructure or reporting error | Failure | None |

## Configuration and trust

- Requires repository secrets `SONAR_HOST_URL` and `SONAR_TOKEN`, plus variable
  `SONAR_PROJECT_KEY`. The project's quality gate must reject new High/Blocker issues.
- `SONAR_BUILD_JAVA_VERSION` optionally overrides hosted Java 17; scanner tooling
  comes from the internal runner image.
- Compilation restores the copied PR's Maven cache first, then the shared main
  baseline, and saves only to that PR branch. Misses download dependencies normally;
  no Sonar warmup or schedule is needed. Other Sonar jobs have cache access disabled.

Approval covers the copied workflow and helpers. Maven runs on a disposable hosted
runner without Sonar credentials. Fresh hosted preflight rejects source symlinks
and unsafe class artifacts before internal scanning, which reads source/classes as
data with step-scoped credentials. Preflight is heuristic; runner labels do not
restrict access from other workflows.

The workflow rechecks the PR head and target before scanning and passing the gate.
Same-PR scans serialize through collection without cancelling active scans.
Validate the workflow before making **Sonar check gate** required.
