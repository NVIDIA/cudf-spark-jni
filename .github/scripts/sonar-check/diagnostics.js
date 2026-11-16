// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const jobNames = {
  select: 'PR source selection',
  security: 'script security preflight',
  scan: 'analysis and collection',
};
const results = new Set(['success', 'failure', 'cancelled', 'skipped']);

module.exports = async function diagnostics({ context, core, env = process.env }) {
  const jobs = JSON.parse(env.JOB_RESULTS);
  if (!jobs || Array.isArray(jobs) || typeof jobs !== 'object') {
    throw new Error('Sonar prerequisite diagnostics require the workflow job results.');
  }
  const unselected = jobs.select?.result === 'success' &&
    jobs.select.outputs?.['run-scan'] === 'false';
  const failed = [];
  const skipped = [];
  for (const [id, job] of Object.entries(jobs)) {
    if (!Object.hasOwn(jobNames, id)) {
      continue;
    }
    if (!job || !results.has(job.result)) {
      throw new Error(`Sonar prerequisite ${id} has no valid job result.`);
    }
    if (job.result === 'failure' || job.result === 'cancelled') {
      failed.push(`${jobNames[id]} ${job.result === 'failure' ? 'failed' : 'was cancelled'}`);
    } else if (job.result === 'skipped') {
      skipped.push(`${jobNames[id]} was skipped`);
    }
  }
  const problems = failed.length ? failed : unselected ? [] : skipped;
  if (!problems.length) {
    return;
  }
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  const run = String(context.runId);
  const attempt = env.GITHUB_RUN_ATTEMPT;
  if (
    repository !== 'NVIDIA/cudf-spark-jni' ||
    !/^[1-9][0-9]*$/.test(run) ||
    typeof attempt !== 'string' ||
    !/^[1-9][0-9]*$/.test(attempt)
  ) {
    throw new Error('Cannot identify the Sonar workflow attempt for prerequisite diagnostics.');
  }
  const url = `https://github.com/${repository}/actions/runs/${run}/attempts/${attempt}`;
  core.setFailed(
    `Sonar prerequisites: ${problems.join('; ')}. Inspect the affected job and rerun after resolving its error: ${url}`,
  );
};
