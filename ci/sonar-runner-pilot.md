# Sonar runner pilot

This test-only workflow runs the existing Java compilation and Sonar analysis on
an ephemeral CPU runner. It does not replace the production Blossom checks.

The workflow runs only when a maintainer applies the `sonar-runner-pilot` label to
an open, same-repository PR targeting `main` or `release/*`. The approved commit
is the PR head at that label event. Opening or updating a PR does not start the
pilot. Remove and reapply the label after reviewing any new head. Fork PRs are
outside this pilot.

## Configuration

| Repository setting | Purpose |
| --- | --- |
| Variable `SONAR_RUNNER_PILOT_LABEL` | Dedicated runner label beginning with `sonar-runner-pilot-`; do not use the production `blossom` label. |
| Variable `SONAR_RUNNER_PILOT_GROUP` | Optional organization runner group. Leave unset for the initial repository runner. |
| Variable `SONAR_PROJECT_KEY` | Existing Sonar project receiving the PR analysis. |
| Secret `SONAR_HOST_URL` | Sonar service URL. |
| Secret `SONAR_TOKEN` | Token with analysis access to that project. |

The worker must provide Linux x86-64, Git, JDK 17, Maven, and SonarScanner CLI 6.2.1
or a compatible approved version. It needs access to GitHub, public Maven
dependencies, and the configured Sonar service. Register each worker for one job
and dispose of its filesystem afterward. The workflow does not receive runner
registration credentials or artifact publication credentials.

## Later validation

Publish the test branch in the same repository and open a draft PR. Keep the
pilot label absent while reviewing the workflow and runner launcher. Because the
workflow listens for a `pull_request` label event, it can be exercised from that
unmerged PR; a new `workflow_dispatch` workflow would require default-branch
activation first.

After validation is authorized:

1. Configure the settings above and start the bounded ephemeral worker pool.
2. Review the exact PR head and apply `sonar-runner-pilot` as a maintainer.
3. Confirm the scan checked out that SHA and the `sonar-runner-pilot` commit
   status links to this workflow run. Success requires Sonar processing and its
   quality gate to pass; an upload alone is insufficient.
4. Confirm the worker deregistered after the job and its environment was removed.

The scan job has a 45-minute execution timeout, including up to 10 minutes for
Sonar processing. The launcher must budget additional startup and idle time and
must not terminate a busy worker on a shorter timer. A missing worker leaves the
job queued; GitHub job execution timeout does not bound that queue wait. Cancel
the workflow explicitly if no worker becomes available during the pilot.

The result is published on the approved SHA even if the PR later advances.
Cancellation or a skipped/failed scan cannot produce a successful pilot status.
The status is supplementary and should not be made a required merge check during
this test-only pilot.
