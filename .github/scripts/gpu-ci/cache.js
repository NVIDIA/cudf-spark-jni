// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

// A cache hit alone may be from a failed or partially uploaded build. Reuse only
// a complete pair from a successful trusted worker, with no intervening build changes.
async function alreadyPrepared({github, context, core, source, pomHash, selected, family,
  catalog, selectSuites, deadline}) {
  const ref = 'refs/heads/main';
  const arch = selected.platform === 'linux/amd64' ? 'X64' : 'ARM64';
  const nativePrefix = `jni-native-v2-Linux-${arch}-${selected['release-line']}-${family}-${selected.version}-`;
  const mavenPrefix = `jni-maven-Linux-${arch}-${pomHash}-`;
  const present = (entry, key) => entry.ref === ref && entry.size_in_bytes > 0 && entry.key === key;
  const read = (method, params) => {
    const remaining = deadline - Date.now();
    if (!(remaining > 0)) throw new Error('Cache lookup deadline exhausted');
    return method({...params, request: {signal: AbortSignal.timeout(Math.min(5000, remaining))}});
  };
  try {
    const {data: native} = await read(github.rest.actions.getActionsCacheList, {
      ...context.repo, ref, key: nativePrefix, per_page: 100,
      sort: 'created_at', direction: 'desc',
    });
    // Restore prefers the newest matching archive. Do not skip a newer partial
    // upload and declare an older complete native cache ready instead.
    const entry = native.actions_caches.find(cache =>
      cache.ref === ref && cache.key.startsWith(nativePrefix));
    if (!entry || !present(entry, entry.key)) return false;
    const match = /^([a-f0-9]{40})-([1-9][0-9]*)-([1-9][0-9]*)$/.exec(
      entry.key.slice(nativePrefix.length));
    if (!match) return false;
    const [, baseline, runId, attempt] = match;
    const mavenKey = `${mavenPrefix}${baseline}-${runId}-${attempt}`;
    const {data: maven} = await read(github.rest.actions.getActionsCacheList, {
      ...context.repo, ref, key: mavenKey, per_page: 100,
    });
    // Maven is shared across CUDA families; the paired archive need not be newest.
    if (!maven.actions_caches.some(cache => present(cache, mavenKey))) return false;
    const {data: run} = await read(github.rest.actions.getWorkflowRunAttempt, {
      ...context.repo, run_id: Number(runId), attempt_number: Number(attempt),
    });
    const repository = `${context.repo.owner}/${context.repo.repo}`;
    if (run.id !== Number(runId) || run.run_attempt !== Number(attempt) ||
        run.status !== 'completed' || run.conclusion !== 'success' ||
        run.event !== 'workflow_dispatch' || run.head_branch !== 'main' ||
        run.path !== '.github/workflows/gpu-ci-worker.yml' ||
        run.repository?.full_name !== repository || run.head_repository?.full_name !== repository) {
      return false;
    }
    if (baseline !== source) {
      const {data: diff} = await read(github.rest.repos.compareCommitsWithBasehead, {
        ...context.repo, basehead: `${baseline}...${source}`, per_page: 1,
      });
      if (diff.status !== 'ahead' || diff.merge_base_commit?.sha !== baseline ||
          !Array.isArray(diff.files) || diff.files.length >= 300) return false;
      const warmerChanged = diff.files.some(file => [file.filename, file.previous_filename]
        .some(path => ['.github/workflows/gpu-cache.yml', '.github/scripts/gpu-ci/cache.js'].includes(path)));
      if (warmerChanged || selectSuites(diff.files, catalog).includes('jni')) return false;
    }
    // Release source history does not include changes to the main-owned worker.
    const path = '.github/workflows/gpu-ci-worker.yml';
    const {data: previousWorker} = await read(github.rest.repos.getContent, {
      ...context.repo, path, ref: run.head_sha,
    });
    const {data: currentWorker} = await read(github.rest.repos.getContent, {
      ...context.repo, path, ref: 'refs/heads/main',
    });
    if (previousWorker.type !== 'file' || currentWorker.type !== 'file' ||
        !previousWorker.sha || previousWorker.sha !== currentWorker.sha) return false;
    core.info(`Cache already prepared by ${run.html_url}: ${baseline}; no build changes through ${source}`);
    return true;
  } catch (error) {
    core.warning(`Cannot confirm a complete warm cache (${error.status || error.name}); prepare it again`);
  }
  return false;
}

module.exports = {alreadyPrepared};
