// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const repository = 'NVIDIA/cudf-spark-jni';
const producerPath = '.github/workflows/sonar-check.yml';
const consumerRef = `${repository}/.github/workflows/sonar-scan.yml@refs/heads/main`;
const supportedBases = new Set(['main', 'release/26.12']);
const activeStatuses = new Set(['queued', 'requested', 'waiting', 'pending', 'in_progress']);
const conclusions = new Set([
  'success', 'failure', 'cancelled', 'skipped', 'neutral', 'timed_out',
  'action_required', 'stale', 'startup_failure',
]);
const shaPattern = /^[0-9a-fA-F]{40}$/;

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function positive(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function sameSource(run, event, workflow) {
  requireValue(
    run && positive(run.run_attempt) && positive(run.run_number) && positive(run.workflow_id) &&
      typeof run.head_sha === 'string' && shaPattern.test(run.head_sha),
    `Sonar producer ${event.id} has malformed attempt, run number, workflow ID, or head SHA.`,
  );
  for (const [field, observed, expected] of [
    ['id', run.id, event.id],
    ['run number', run.run_number, event.run_number],
    ['workflow ID', run.workflow_id, event.workflow_id],
    ['registered workflow ID', run.workflow_id, workflow.id],
    ['path', run.path, producerPath],
    ['name', run.name, 'Sonar preparation'],
    ['event', run.event, 'pull_request'],
    ['repository', run.repository?.full_name, repository],
    ['head SHA', run.head_sha.toLowerCase(), event.head_sha.toLowerCase()],
    ['head branch', run.head_branch, event.head_branch],
    ['head repository', run.head_repository?.full_name, event.head_repository.full_name],
  ]) {
    requireValue(
      observed === expected,
      `Sonar producer ${event.id} ${field} mismatch: observed ${JSON.stringify(observed)}, expected ${JSON.stringify(expected)}.`,
    );
  }
  requireValue(
    run.run_attempt >= event.run_attempt,
    `Sonar producer attempt is older than the event: observed ${run.run_attempt}, expected ${event.run_attempt}.`,
  );
}

async function currentPR(github, context, request, run) {
  const numbers = new Set();
  const owner = run.head_repository.full_name.split('/')[0];
  for (let page = 1; page <= 10; page++) {
    const { data } = await github.rest.pulls.list({
      ...context.repo, state: 'open', head: `${owner}:${run.head_branch}`,
      per_page: 100, page, request: request(),
    });
    requireValue(Array.isArray(data) && data.length <= 100, 'GitHub returned invalid producer PR candidates.');
    for (const pr of data) {
      requireValue(
        positive(pr?.number) && pr.base?.repo?.full_name === repository && !numbers.has(pr.number),
        'GitHub returned invalid, foreign, or duplicate producer PR candidates.',
      );
      numbers.add(pr.number);
    }
    if (data.length < 100) break;
    requireValue(page < 10, 'Producer PR candidates exceed the 1000-entry observation limit.');
  }
  const matches = [];
  for (const number of numbers) {
    const { data: pr } = await github.rest.pulls.get({
      ...context.repo, pull_number: number, request: request(),
    });
    requireValue(
      pr?.number === number && pr.base?.repo?.full_name === repository &&
        ['open', 'closed'].includes(pr.state),
      'GitHub returned invalid current PR metadata for the Sonar producer.',
    );
    if (pr.state === 'closed') continue;
    requireValue(
      typeof pr.head?.sha === 'string' && shaPattern.test(pr.head.sha) &&
        typeof pr.head.ref === 'string' && typeof pr.base.ref === 'string',
      'GitHub returned incomplete current PR source metadata for the Sonar producer.',
    );
    if (
      supportedBases.has(pr.base.ref) && pr.head.sha.toLowerCase() === run.head_sha.toLowerCase() &&
      pr.head.ref === run.head_branch && pr.head.repo?.full_name === run.head_repository.full_name
    ) {
      matches.push(pr);
    }
  }
  requireValue(matches.length <= 1, 'Multiple open PRs match this Sonar producer; its target is ambiguous.');
  return { pr: matches[0] || null, empty: numbers.size === 0 };
}

module.exports = async function resolveSource({ github, context, env = process.env, request }) {
  const event = context.payload.workflow_run;
  requireValue(
    context.eventName === 'workflow_run' &&
      context.payload.action === 'completed' &&
      `${context.repo.owner}/${context.repo.repo}` === repository &&
      context.payload.repository?.full_name === repository &&
      env.GITHUB_REPOSITORY === repository && env.GITHUB_WORKFLOW_REF === consumerRef &&
      positive(event?.id) && positive(event.run_attempt) && positive(event.run_number) &&
      positive(event.workflow_id) && typeof event.head_sha === 'string' && shaPattern.test(event.head_sha) &&
      typeof event.head_branch === 'string' && event.head_branch.length > 0 &&
      typeof event.head_repository?.full_name === 'string' && event.head_repository.full_name.length > 0 &&
      typeof request === 'function',
    'Sonar source resolution requires the trusted main consumer and an exact preparation run event.',
  );
  const visibilityDeadline = Date.now() + 60000;
  const observe = () => {
    const options = request();
    const remaining = visibilityDeadline - Date.now();
    requireValue(remaining > 0, 'Sonar producer visibility exceeded its initial 60-second window.');
    return { ...options, timeout: Math.min(options.timeout, remaining) };
  };
  const pause = async () => {
    const options = request();
    const remaining = Math.min(visibilityDeadline - Date.now(), options.timeout);
    if (remaining <= 0) return false;
    await new Promise(resolve => setTimeout(resolve, Math.min(2000, remaining)));
    return Date.now() < visibilityDeadline;
  };
  const { data: workflow } = await github.rest.actions.getWorkflow({
    ...context.repo, workflow_id: 'sonar-check.yml', request: request(),
  });
  requireValue(
    positive(workflow?.id) && workflow.path === producerPath && workflow.name === 'Sonar preparation',
    'The registered producer must be the Sonar preparation workflow at .github/workflows/sonar-check.yml.',
  );
  let run;
  for (;;) {
    try {
      ({ data: run } = await github.rest.actions.getWorkflowRun({
        ...context.repo, run_id: event.id, request: observe(),
      }));
    } catch (error) {
      if (error.status !== 404) throw error;
      if (await pause()) continue;
      throw new Error(`Sonar producer ${event.id} remained unavailable (HTTP 404) for 60 seconds.`, { cause: error });
    }
    sameSource(run, event, workflow);
    if (run.run_attempt > event.run_attempt) return { pr: null, run, stale: true };
    requireValue(
      Object.hasOwn(run, 'conclusion') &&
        (run.status === 'completed' || activeStatuses.has(run.status)) &&
        (run.conclusion === null || conclusions.has(run.conclusion)) &&
        (run.status === 'completed' || run.conclusion === null),
      `Sonar producer has invalid lifecycle metadata: status=${run.status}, conclusion=${run.conclusion}.`,
    );
    const complete = run.status === 'completed' && run.conclusion !== null;
    if (complete) break;
    if (!(await pause())) {
      throw new Error(
        `Sonar producer ${event.id} did not reach ${context.payload.action} visibility within 60 seconds; ` +
        `observed status=${run.status}, conclusion=${run.conclusion}.`,
      );
    }
  }
  // Fork runs can omit pull_requests, and their commit may have no association
  // in the base repository. Resolve the head there, then verify the exact tuple.
  for (;;) {
    const candidate = await currentPR(github, context, observe, run);
    if (candidate.pr) return { pr: candidate.pr, run, stale: false };
    if (!candidate.empty || !(await pause())) return { pr: null, run, stale: true };
  }
};
