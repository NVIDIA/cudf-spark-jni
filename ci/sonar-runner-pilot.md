# Sonar runner pilot

Test-only validation on dummy PR #5400. Do not merge the deliberate Java findings.
A maintainer applies `sonar-runner-pilot` to approve the current same-repository
head. New commits require a new label event; ordinary pushes do not run the pilot.

The workflow validates common PR #77 at the reviewed commit
`b0cb860cb186d6bae817229270f58c8f2bfe94ff` with this path:

1. A hosted job scans the exact approved source without executing its build.
2. A `spark-sonar-internal` worker compiles Java without Sonar credentials.
3. A fresh worker downloads that build's exact artifact, verifies its digest,
   and runs Sonar and the common collector. It does not execute Maven/plugins.
4. The hosted common reporter posts its own summary and **Sonar check** using
   the workflow token. New High/Blocker issues, a failed gate, or missing evidence
   fail the check. The intentional example should demonstrate a failure.

The common scanner image is not published yet. This unmerged pilot explicitly
builds the reviewed image recipe on the hosted worker and invokes the same scan
core by immutable local image ID. This validates the proposed image and source
policy; it does not validate the public `scan-scripts` action's registry pull.
Production consumers must use that action with the published prebuilt image.

## Settings

- Variable `SONAR_PROJECT_KEY`: existing Sonar project.
- Variable `SONAR_RUNNER_PILOT_MAVEN_MIRROR`: anonymous HTTPS Maven mirror.
- Existing secrets `SONAR_HOST_URL` and `SONAR_TOKEN`: scanner/analysis-read access,
  including scanner-context permission for exact source identity verification.

Runner administration credentials remain in Jenkins. The workflow only selects
`[self-hosted, Linux, X64, spark-sonar-internal]`; it creates no runners or jobs.
The pool replenishes each one-job worker after completion. Compilation and scan
have separate 30-minute execution limits; queue waiting needs operator monitoring.
The reporter always runs after an approved attempt, including preflight failure.

Artifacts are bound to the producing run/attempt and downloaded by exact artifact
ID with digest mismatches treated as errors. Keep the PR label absent after testing
and remove only pilot-created variables; retain existing secrets and the shared
runner pool. The check remains supplementary during validation.
