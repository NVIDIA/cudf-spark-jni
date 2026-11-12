# Sonar runner pilot

Test-only validation on dummy PR #5433. Do not merge the deliberate Java findings.
A maintainer applies `sonar-runner-pilot` to approve the current same-repository
head. New commits require a new label event; ordinary pushes do not run the pilot.
The workflow serializes each PR through Sonar processing and result collection.
Different PRs may scan concurrently in production; this pilot allows only #5433.

The workflow validates common PR #77 at the reviewed commit
`2c257f7386ee0d5ac80f6a3510f022ba1024b815` with this path:

1. A hosted job scans the exact approved source without executing its build.
2. A `spark-sonar-internal` worker compiles Java without Sonar credentials.
3. A fresh worker downloads that build's exact artifact, verifies its digest,
   and runs Sonar and the common collector. It does not execute Maven or PR build scripts.
4. The hosted common reporter publishes the scoped **Sonar check** using
   the workflow token. Clean passes post no comment; findings receive a summary.
   New High/Blocker issues, a failed gate, or missing evidence
   fail the check. The earlier division-by-zero example validates failure; the
   repaired version retains only a Medium finding to validate a passing check.

The shared reporter sets `Sonar check / PR #N / <scope>` pending after approval
and updates that same context after scanning. Infrastructure/process failures
fail the check with retry guidance and do not post PR comments. The scope binds the Sonar server,
project, and target branch, keeping unrelated Sonar checks independent.

The common scanner image is not published yet. This unmerged pilot explicitly
builds the reviewed image recipe on the hosted worker and invokes the same scan
core by immutable local image ID. This validates the proposed image and source
policy; it does not validate the public `scan-scripts` action's registry pull.
Production consumers must use that action with the published prebuilt image.

## Settings

- Variable `SONAR_PROJECT_KEY`: existing Sonar project.
- Variable `SONAR_RUNNER_PILOT_MAVEN_MIRROR`: anonymous HTTPS Maven mirror.
- Existing secret `SONAR_HOST_URL`: Sonar service URL.
- Temporary `SONAR_RUNNER_PILOT_TOKEN`: the existing Jenkins Sonar credential,
  securely transferred only for validation. The repository's `SONAR_TOKEN` is
  rejected by the current service and is left unchanged. The credential must allow
  scanner-context reads for exact source identity verification.

Runner administration credentials remain in Jenkins. The workflow only selects
`[self-hosted, Linux, X64, spark-sonar-internal]`; it creates no runners or jobs.
The pool replenishes each one-job worker after completion. Compilation and scan
have separate 30-minute execution limits; queue waiting needs operator monitoring.
The reporter always runs after an approved attempt, including preflight failure.

Artifacts are bound to the producing run/attempt and downloaded by exact artifact
ID with digest mismatches treated as errors. Keep the PR label absent after testing
and remove the pilot-created variables and temporary secret; retain existing
secrets and the shared runner pool. The check remains supplementary during validation.
