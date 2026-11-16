// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const checkName = 'Sonar check gate';

function result(jobs) {
  if (jobs.select?.result !== 'success') {
    return { conclusion: 'failure', message: 'Sonar file selection or gate initialization failed.' };
  }
  const selected = jobs.select.outputs?.['run-scan'];
  if (selected === 'false') {
    const reason = jobs.select.outputs.reason;
    if (!['title-skip', 'no-code-changes'].includes(reason)) {
      throw new Error('Sonar selection omitted its skip reason.');
    }
    if (['security', 'compile', 'scan', 'report'].some(id =>
      !['success', 'skipped'].includes(jobs[id]?.result))) {
      return { conclusion: 'failure', message: 'An unselected Sonar job failed or has no valid result.' };
    }
    return {
      conclusion: 'skipped',
      message: reason === 'title-skip'
        ? 'Skipped because the PR title requests CI to be skipped.'
        : 'Skipped because the PR changes no files analyzed by Sonar.',
    };
  }
  if (selected !== 'true') throw new Error('Sonar selection did not return a valid decision.');
  const names = {
    security: 'script preflight',
    compile: 'Java compilation',
    scan: 'internal analysis',
    report: 'result reporting',
  };
  const failed = Object.entries(names).filter(([id]) => jobs[id]?.result !== 'success');
  return failed.length
    ? {
      conclusion: 'failure',
      message: `Sonar failed: ${failed.map(([id, name]) => `${name} ${jobs[id]?.result || 'missing'}`).join('; ')}. See the workflow logs for the exact error.`,
    }
    : { conclusion: 'success', message: 'Sonar analysis and result publication passed.' };
}

async function gate({ github, context, core, initial = false, env = process.env }) {
  const pr = context.payload.pull_request;
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  const sha = pr?.head?.sha;
  const attempt = Number(env.GITHUB_RUN_ATTEMPT);
  const runNumber = Number(env.GITHUB_RUN_NUMBER);
  if (
    repository !== 'NVIDIA/cudf-spark-jni' ||
    context.eventName !== 'pull_request_target' ||
    pr?.base?.repo?.full_name !== repository || !['main', 'release/26.12'].includes(pr.base.ref) ||
    !Number.isSafeInteger(pr.number) || pr.number <= 0 || !/^[0-9a-f]{40}$/.test(sha || '') ||
    !Number.isSafeInteger(context.runId) || context.runId <= 0 ||
    !Number.isSafeInteger(attempt) || attempt <= 0 ||
    !Number.isSafeInteger(runNumber) || runNumber <= 0 ||
    env.GITHUB_WORKFLOW_REF !== `${repository}/.github/workflows/sonar-check.yml@refs/heads/main`
  ) {
    throw new Error('Sonar gate requires the trusted trigger and an exact main or release/26.12 PR head.');
  }
  let jobs;
  let resultError;
  if (!initial) {
    try {
      jobs = JSON.parse(env.JOB_RESULTS);
    } catch (error) {
      resultError = error;
    }
  }
  // A failed-job rerun retains outputs from the successful selection job. Its
  // gate belongs to the producing attempt, not necessarily this final job's.
  const gateAttempt = Number(jobs?.select?.outputs?.['gate-attempt'] || attempt);
  if (!Number.isSafeInteger(gateAttempt) || gateAttempt <= 0 || gateAttempt > attempt) {
    throw new Error('Sonar gate has an invalid producing attempt.');
  }
  const prefix = `sonar:${repository}:${pr.number}:`;
  const externalId = `${prefix}${runNumber}:${context.runId}:${gateAttempt}`;
  const detailsUrl = `https://github.com/${repository}/actions/runs/${context.runId}/attempts/${attempt}`;
  const deadline = Date.now() + 120000;
  const common = (limit = deadline) => {
    const remaining = Math.min(deadline, limit) - Date.now();
    if (remaining <= 0) throw new Error('Sonar gate exceeded its API deadline before the next request.');
    return { ...context.repo, request: { timeout: Math.min(10000, remaining) } };
  };
  // The select and final gate jobs share a short, FIFO concurrency group. Do not
  // serialize the scan itself here, or a title edit could not reset the gate.
  const visibilityDeadline = Math.min(deadline, Date.now() + 60000);
  let matches;
  let owned;
  for (;;) {
    if (!initial && matches !== undefined && Date.now() >= visibilityDeadline) break;
    const checks = [];
    for (let page = 1; page <= 10; page++) {
      const { data } = await github.rest.checks.listForRef({
        ...common(initial ? deadline : visibilityDeadline),
        ref: sha, check_name: checkName, filter: 'all', per_page: 100, page,
      });
      if (!Number.isSafeInteger(data?.total_count) || !Array.isArray(data.check_runs) ||
          data.total_count > 1000 || data.check_runs.length > 100) {
        throw new Error('Cannot verify Sonar gate ownership: invalid or excessive check history.');
      }
      checks.push(...data.check_runs);
      if (checks.length === data.total_count) break;
      if (data.check_runs.length !== 100 || checks.length > data.total_count || page === 10) {
        throw new Error('Cannot verify Sonar gate ownership: truncated check history.');
      }
    }
    owned = checks.filter(check => check.name === checkName && check.head_sha === sha &&
      check.app?.slug === 'github-actions' && typeof check.external_id === 'string' &&
      check.external_id.startsWith(prefix));
    let newer = false;
    for (const check of owned) {
      const identity = check.external_id.slice(prefix.length).match(/^([1-9][0-9]*):([1-9][0-9]*):([1-9][0-9]*)$/);
      if (!identity || !Number.isSafeInteger(check.id) || check.id <= 0) {
        throw new Error('Cannot verify Sonar gate ownership: malformed check identity.');
      }
      newer ||= BigInt(identity[1]) > BigInt(runNumber) ||
        (BigInt(identity[1]) === BigInt(runNumber) && BigInt(identity[3]) > BigInt(gateAttempt));
    }
    if (newer) {
      core.setOutput('active', 'false');
      core.notice('A newer Sonar workflow owns this commit gate; leaving its result untouched.');
      return;
    }
    matches = owned.filter(check => check.external_id === externalId);
    if (matches.length > 1) throw new Error('Multiple checks claim this Sonar run and attempt.');
    if (initial || matches.length || Date.now() >= visibilityDeadline) break;
    // Only initial check visibility gets grace. Never repeat the create or
    // weaken identity checks while reconciling a lost creation response.
    await new Promise(resolve => setTimeout(resolve,
      Math.min(2000, visibilityDeadline - Date.now())));
  }
  const { data: current } = await github.rest.pulls.get({
    ...common(), pull_number: pr.number,
  });
  if (current?.number !== pr.number || typeof current.title !== 'string' ||
      !['open', 'closed'].includes(current.state) || !current.head?.sha || !current.base?.ref) {
    throw new Error('Cannot verify Sonar gate policy: invalid current PR response.');
  }
  const sameRevision = current.state === 'open' && current.head.sha === sha &&
    current.base.ref === pr.base.ref && current.base.repo?.full_name === repository;
  if (initial) {
    if (!sameRevision) {
      core.setOutput('active', 'false');
      core.notice('The PR revision or target changed; its current event owns the Sonar gate.');
      return;
    }
    if (matches.length) throw new Error('This Sonar run already initialized its gate; refusing a duplicate write.');
    // Retire only superseded check records, not their workflows or scans. Do
    // this before creating the replacement so a late completion cannot become
    // the newest required result or leave an obsolete pending gate behind.
    for (const previous of owned) {
      if (previous.status !== 'completed') {
        await github.rest.checks.update({
          ...common(), check_run_id: previous.id, status: 'completed', conclusion: 'cancelled',
          output: { title: checkName, summary: 'Superseded by a newer Sonar workflow for this commit.' },
        });
      }
    }
    // A lost response may hide an accepted write. The final job can reconcile
    // this exact external_id; never replay the create automatically.
    const { data: check } = await github.rest.checks.create({
      ...common(), name: checkName, head_sha: sha, external_id: externalId,
      details_url: detailsUrl, status: 'in_progress',
    });
    if (!Number.isSafeInteger(check?.id) || check.id <= 0) {
      throw new Error('GitHub did not return the initialized Sonar gate ID.');
    }
    core.setOutput('check-id', String(check.id));
    core.setOutput('gate-attempt', String(attempt));
    core.setOutput('active', 'true');
    return;
  }
  const check = matches[0];
  if (!check) {
    if (!sameRevision) return;
    throw new Error('Sonar gate was not visible after 60 seconds; no create was replayed. Rerun the workflow.');
  }
  if (check.status === 'completed' && check.conclusion === 'cancelled') {
    core.notice('This Sonar gate was cancelled or superseded; leaving its terminal result untouched.');
    return;
  }
  let outcome;
  try {
    if (resultError) throw resultError;
    const selectedId = jobs.select?.outputs?.['gate-id'];
    if (selectedId && selectedId !== String(check.id)) {
      throw new Error('The selected gate ID differs from this run-owned check.');
    }
    outcome = result(jobs);
    if (!sameRevision) {
      outcome = { conclusion: 'cancelled', message: 'The PR revision or target changed before completion.' };
    } else if (/\[skip ci\]/.test(current.title)) {
      outcome = { conclusion: 'skipped', message: 'The current PR title requests CI to be skipped.' };
    } else if (jobs.select?.outputs?.reason === 'title-skip') {
      outcome = { conclusion: 'failure', message: 'The CI-skip title marker was removed; the newer workflow must run Sonar.' };
    }
  } catch (error) {
    core.error(`Cannot evaluate Sonar job results: ${error.message}`);
    outcome = { conclusion: 'failure', message: `Cannot evaluate Sonar gate: ${error.message}` };
  }
  await github.rest.checks.update({
    ...common(), check_run_id: check.id, status: 'completed', conclusion: outcome.conclusion,
    ...(outcome.conclusion === 'success' ? {} : {
      output: { title: checkName, summary: `${outcome.message}\n\n[Workflow logs](${detailsUrl})` },
    }),
  });
  if (outcome.conclusion === 'failure') core.setFailed(outcome.message);
}

module.exports = gate;
module.exports.result = result;
