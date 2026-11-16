// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const resolveSource = require('./source.js');

module.exports = async function prepare({ github, context, core, env = process.env }) {
  const deadline = Date.now() + 120000;
  const request = () => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error('Sonar artifact selection exceeded its 120-second API budget.');
    return { timeout: Math.min(10000, remaining) };
  };
  const { pr, run, stale } = await resolveSource({ github, context, env, request });
  if (stale || pr.head.sha !== env.APPROVED_SHA || String(pr.number) !== env.PR_NUMBER ||
      String(run.id) !== env.SOURCE_RUN_ID || String(run.run_attempt) !== env.SOURCE_ATTEMPT) {
    throw new Error('The preparation run or PR changed before artifact selection.');
  }
  if (run.status !== 'completed' || run.conclusion !== 'success') {
    throw new Error(`Hosted Sonar preparation is ${run.status}/${run.conclusion}; inspect run ${run.id}, attempt ${run.run_attempt}.`);
  }
  if (!/^org\/[A-Za-z0-9][A-Za-z0-9_. -]{0,99}$/.test(env.SONAR_RUNNER_GROUP || '')) {
    throw new Error('Configure SONAR_RUNNER_GROUP as org/<name> after restricting that organization group to the trusted sonar-scan workflow.');
  }

  const artifacts = [];
  for (let page = 1; page <= 10; page++) {
    const { data } = await github.rest.actions.listWorkflowRunArtifacts({
      ...context.repo, run_id: run.id, per_page: 100, page, request: request(),
    });
    if (!Number.isSafeInteger(data?.total_count) || data.total_count < 0 ||
        data.total_count > 1000 || !Array.isArray(data.artifacts) || data.artifacts.length > 100) {
      throw new Error('GitHub returned invalid artifact metadata for Sonar preparation.');
    }
    artifacts.push(...data.artifacts);
    if (artifacts.length === data.total_count) break;
    if (data.artifacts.length !== 100 || artifacts.length > data.total_count || page === 10) {
      throw new Error('The preparation artifact list is incomplete.');
    }
  }
  // A partial rerun may reuse an earlier successful compile from the same run
  // and exact head. Choose its newest retained attempt, never another run's data.
  const candidates = artifacts.flatMap(artifact => {
    const match = /^sonar-classes-([1-9][0-9]*)$/.exec(artifact.name || '');
    if (!match || artifact.expired === true) return [];
    const attempt = Number(match[1]);
    if (!Number.isSafeInteger(attempt) || attempt > run.run_attempt) {
      throw new Error('A Sonar class artifact claims an invalid or future preparation attempt.');
    }
    return [{ artifact, attempt }];
  });
  const latest = Math.max(...candidates.map(item => item.attempt));
  const matches = candidates.filter(item => item.attempt === latest);
  if (matches.length !== 1) throw new Error('No unique retained class artifact exists for this preparation run.');
  const artifact = matches[0].artifact;
  if (!Number.isSafeInteger(artifact.id) || artifact.id <= 0 || artifact.expired !== false ||
      !Number.isSafeInteger(artifact.size_in_bytes) || artifact.size_in_bytes <= 0 ||
      artifact.size_in_bytes > 512 * 1024 * 1024 || !/^sha256:[0-9a-f]{64}$/.test(artifact.digest || '')) {
    throw new Error('The Sonar class artifact has invalid identity, digest or size (maximum 512 MiB).');
  }
  if (Object.hasOwn(artifact, 'workflow_run') &&
      (artifact.workflow_run?.id !== run.id || artifact.workflow_run.head_sha !== pr.head.sha)) {
    throw new Error('The Sonar class artifact belongs to a different preparation run or commit.');
  }
  core.setOutput('artifact-id', String(artifact.id));
};
