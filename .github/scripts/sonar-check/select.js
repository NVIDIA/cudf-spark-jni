// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const repository = 'NVIDIA/cudf-spark-jni';
const supportedBases = new Set(['main', 'release/26.12']);
const shaPattern = /^[0-9a-fA-F]{40}$/;
const fileStatuses = new Set([
  'added', 'removed', 'modified', 'renamed', 'copied', 'changed', 'unchanged',
]);
const approvedRoot = path.resolve(__dirname, '../../..');

function requireValue(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function validPath(value) {
  return typeof value === 'string' && value.length > 0 &&
    !value.startsWith('/') && !value.includes('\0');
}

function validPattern(value) {
  return validPath(value) && !/[?\[\]{}()!\\]/.test(value) &&
    value.split('/').every(part => part.length > 0 && part !== '.' && part !== '..' &&
      (!part.includes('**') || part === '**'));
}

function validatePR(pr, expected) {
  requireValue(
    pr?.number === expected.number &&
      pr.state === 'open' &&
      pr.base?.repo?.full_name === repository &&
      pr.base.ref === expected.base.ref &&
      typeof pr.base.sha === 'string' &&
      shaPattern.test(pr.base.sha) &&
      typeof pr.head?.sha === 'string' &&
      pr.head.sha.toLowerCase() === expected.head.sha.toLowerCase() &&
      pr.head.ref === expected.head.ref &&
      typeof pr.head.ref === 'string' &&
      pr.head.ref.length > 0 &&
      !/\s|[\x00-\x1f\x7f]/.test(pr.head.ref) &&
      typeof pr.head.repo?.full_name === 'string' &&
      pr.head.repo.full_name === expected.head.repo?.full_name &&
      typeof pr.title === 'string',
    `PR #${expected.number} is closed, changed, or has invalid metadata: ` +
      `expected head=${expected.head.sha}, target=${expected.base.ref}; ` +
      `observed state=${pr?.state}, head=${pr?.head?.sha}, target=${pr?.base?.ref}. ` +
      'Approve and run the current PR revision.',
  );
  requireValue(
    !pr.head.ref.includes('${'),
    'The PR branch contains ${, which Sonar interprets as property expansion; rename the branch before scanning.',
  );
}

function pathMatcher() {
  const filename = path.join(approvedRoot, '.github/sonar-paths.json');
  const catalog = JSON.parse(fs.readFileSync(filename, 'utf8'));
  const values = name => Object.hasOwn(catalog ?? {}, name) ? catalog[name] : [];
  const extensions = values('extensions');
  const patterns = values('paths');
  const excluded = values('excluded-paths');
  const excludedNames = values('trigger-excluded-filenames');
  const excludedDirectories = values('trigger-excluded-directories');
  requireValue(
    catalog && !Array.isArray(catalog) &&
      [extensions, patterns, excluded, excludedNames, excludedDirectories].every(values => Array.isArray(values) &&
        new Set(values).size === values.length) &&
      extensions.length + patterns.length > 0 &&
      extensions.every(value => typeof value === 'string' && /^[a-z0-9][a-z0-9+]*$/.test(value)) &&
      patterns.every(validPattern) &&
      excluded.every(value => validPath(value) && value.endsWith('/') &&
        !/[?*\[\]{}()!\\]/.test(value)) &&
      excludedNames.every(value => validPattern(value) && !value.includes('/')) &&
      excludedDirectories.every(value => validPattern(value) && !/[/*]/.test(value)),
    'The approved Sonar catalog needs unique extensions, paths, directory prefixes, and trigger-only filename or directory exclusions.',
  );
  const included = new Set(extensions.map(extension => `.${extension}`));
  const skippedDirectories = new Set(excludedDirectories.map(value => value.toLowerCase()));
  return filename => {
    if (typeof filename !== 'string') return false;
    // Trigger-only rules do not reduce the files analyzed in a selected scan.
    // Match names separately so hidden directories receive the same policy.
    const parts = filename.toLowerCase().split('/');
    if (parts.slice(0, -1).some(part => skippedDirectories.has(part)) ||
        excludedNames.some(pattern => path.posix.matchesGlob(parts.at(-1), pattern.toLowerCase()))) {
      return false;
    }
    // Only Sonar's own workflow triggers a scan. Other workflow YAML remains
    // in scanner coverage when supported source or script changes select a scan.
    if (filename.startsWith('.github/workflows/') && /\.ya?ml$/.test(filename) &&
        filename !== '.github/workflows/sonar-check.yml') {
      return false;
    }
    return !excluded.some(prefix => filename.startsWith(prefix)) && (
      included.has(path.posix.extname(filename)) ||
      patterns.some(pattern => path.posix.matchesGlob(filename, pattern))
    );
  };
}

async function changedFiles(github, request, pr) {
  requireValue(
    Number.isSafeInteger(pr.changed_files) && pr.changed_files >= 0 && pr.changed_files <= 3000,
    'The complete PR diff is unavailable: changed_files must be an integer from 0 to 3000.',
  );
  const files = [];
  const seen = new Set();
  for (let page = 1; page <= 30; page++) {
    const { data } = await github.rest.pulls.listFiles({
      owner: 'NVIDIA', repo: 'cudf-spark-jni', pull_number: pr.number,
      per_page: 100, page, request: request(),
    });
    requireValue(Array.isArray(data) && data.length <= 100, 'GitHub returned an invalid PR file-list page.');
    for (const file of data) {
      requireValue(
        file && validPath(file.filename) && fileStatuses.has(file.status) &&
          (!Object.hasOwn(file, 'previous_filename') || validPath(file.previous_filename)) &&
          (file.status !== 'renamed' || validPath(file.previous_filename)),
        'GitHub returned an invalid filename, status, or previous path in the PR diff.',
      );
      requireValue(
        !seen.has(file.filename),
        'The PR file list contains duplicates; complete selection cannot be verified.',
      );
      seen.add(file.filename);
      files.push(file);
    }
    requireValue(
      files.length <= pr.changed_files,
      'The PR file list exceeds changed_files; the diff changed during selection.',
    );
    if (files.length === pr.changed_files) {
      return files;
    }
    requireValue(
      data.length === 100,
      'The PR file list is truncated; refusing to skip an incomplete diff.',
    );
  }
  throw new Error('The complete PR diff was not available within GitHub\'s 3000-file limit.');
}

module.exports = async function select({ github, context, core, initial = false, env = process.env }) {
  // The copied-branch approval covers the whole revision, including its workflow
  // and helpers. These checks validate freshness; they do not grant approval.
  const match = /^refs\/heads\/pull-request\/([1-9][0-9]*)$/.exec(context.ref || '');
  const number = match ? Number(match[1]) : NaN;
  requireValue(
    context.eventName === 'push' && context.payload.deleted === false &&
      `${context.repo.owner}/${context.repo.repo}` === repository &&
      context.payload.repository?.full_name === repository && env.GITHUB_REPOSITORY === repository &&
      Number.isSafeInteger(number) && number > 0 &&
      typeof context.sha === 'string' && shaPattern.test(context.sha) &&
      typeof context.payload.after === 'string' &&
      context.payload.after.toLowerCase() === context.sha.toLowerCase(),
    'Sonar requires a non-deleted push of an exact commit to NVIDIA/cudf-spark-jni pull-request/<number>.',
  );
  const deadline = Date.now() + 120000;
  const request = () => {
    const remaining = deadline - Date.now();
    requireValue(remaining > 0, 'PR selection exceeded its 120-second API observation budget.');
    return { timeout: Math.min(10000, remaining) };
  };
  const getPR = async () => {
    const { data } = await github.rest.pulls.get({
      ...context.repo, pull_number: number, request: request(),
    });
    return data;
  };
  const pr = await getPR();
  const baseRef = initial ? pr?.base?.ref : env.APPROVED_BASE_REF;
  requireValue(
    supportedBases.has(baseRef),
    `Sonar only supports main or release/26.12; observed target ${JSON.stringify(baseRef)}.`,
  );
  const expected = {
    number, base: { ref: baseRef },
    head: { sha: context.sha, ref: pr?.head?.ref, repo: { full_name: pr?.head?.repo?.full_name } },
  };
  validatePR(pr, expected);
  const titleSkip = /\[skip ci\]/.test(pr.title);
  if (!initial) {
    requireValue(
      typeof env.APPROVED_SHA === 'string' &&
        env.APPROVED_SHA.toLowerCase() === context.sha.toLowerCase() && env.PR_NUMBER === String(number),
      'The queued scan does not match the selected PR revision and target.',
    );
    requireValue(!titleSkip, 'The PR now requests CI to be skipped; the queued scan will not run.');
    return;
  }
  let runScan = false;
  let reason = 'title-skip';
  if (!titleSkip) {
    const matches = pathMatcher();
    const files = await changedFiles(github, request, pr);
    const current = await getPR();
    validatePR(current, expected);
    requireValue(
      current.base.sha === pr.base.sha &&
        current.changed_files === pr.changed_files &&
        current.title === pr.title,
      'The PR diff or title changed during selection; use the workflow for the current revision.',
    );
    runScan = files.some(file => matches(file.filename) || matches(file.previous_filename));
    reason = runScan ? 'code-changes' : 'no-code-changes';
  }
  const approvedSha = execFileSync('git', ['-C', approvedRoot, 'rev-parse', '--verify', 'HEAD^{commit}'], {
    encoding: 'utf8', timeout: 10000, maxBuffer: 1024,
  }).trim();
  requireValue(
    shaPattern.test(approvedSha) && approvedSha.toLowerCase() === context.sha.toLowerCase(),
    `The workflow helper checkout must match approved commit ${context.sha}; observed ${approvedSha}.`,
  );
  for (const [name, value] of Object.entries({
    'pr-number': String(pr.number), sha: pr.head.sha.toLowerCase(), branch: pr.head.ref,
    'base-ref': baseRef, 'run-scan': String(runScan), reason,
    'title-skip': String(titleSkip), 'trusted-sha': approvedSha.toLowerCase(),
  })) {
    core.setOutput(name, value);
  }
  core.notice(`Sonar selection: ${reason}.`);
};
