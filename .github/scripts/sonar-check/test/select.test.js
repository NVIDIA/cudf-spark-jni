// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const select = require('../select.js');

// CPU CI may check out a merge commit rather than the contributor's branch head.
const root = path.resolve(__dirname, '../../../..');
const sha = execFileSync('git', ['-C', root, 'rev-parse', '--verify', 'HEAD^{commit}'], {
  encoding: 'utf8', timeout: 10000, maxBuffer: 1024,
}).trim();
const otherSha = sha === 'a'.repeat(40) ? 'b'.repeat(40) : 'a'.repeat(40);
const repository = 'NVIDIA/cudf-spark-jni';
const modified = filename => ({ filename, status: 'modified' });

function fixture(files = [modified('src/Example.java')]) {
  const pr = {
    number: 7, state: 'open', title: 'Update source', changed_files: files.length,
    base: { ref: 'release/26.12', sha: 'c'.repeat(40), repo: { full_name: repository } },
    head: { sha, ref: 'feature/sonar-check', repo: { full_name: 'contributor/cudf-spark-jni' } },
  };
  const outputs = {};
  const calls = { pr: [], files: [] };
  const args = {
    initial: true,
    env: {
      GITHUB_REPOSITORY: repository,
      APPROVED_SHA: sha, APPROVED_BASE_REF: 'release/26.12', PR_NUMBER: '7',
    },
    context: {
      eventName: 'push', ref: 'refs/heads/pull-request/7', sha,
      repo: { owner: 'NVIDIA', repo: 'cudf-spark-jni' },
      payload: { deleted: false, after: sha, repository: { full_name: repository } },
    },
    core: {
      setOutput(name, value) { outputs[name] = value; },
      notice() {},
    },
    github: { rest: { pulls: {
      async get(request) {
        calls.pr.push(request);
        return { data: structuredClone(pr) };
      },
      async listFiles(request) {
        calls.files.push(request);
        const start = (request.page - 1) * request.per_page;
        return { data: structuredClone(files.slice(start, start + request.per_page)) };
      },
    } } },
  };
  return { args, pr, outputs, calls };
}

async function rejected(f, error) {
  await assert.rejects(select(f.args), error);
  assert.deepEqual(f.outputs, {}, 'Rejected observations must not publish a selection.');
}

test('selects the approved source and helper revision for both supported targets', async t => {
  for (const base of ['main', 'release/26.12']) {
    await t.test(base, async () => {
      const f = fixture();
      f.pr.base.ref = base;
      await select(f.args);
      assert.deepEqual(f.outputs, {
        'pr-number': '7', sha, branch: 'feature/sonar-check', 'base-ref': base,
        'run-scan': 'true', reason: 'code-changes', 'title-skip': 'false', 'trusted-sha': sha,
      });
      assert.equal(f.calls.pr.length, 2);
      assert.equal(f.calls.files.length, 1);
      for (const request of [...f.calls.pr, ...f.calls.files]) {
        assert.equal(request.pull_number, 7);
        assert(request.request.timeout > 0 && request.request.timeout <= 10000);
      }
    });
  }
});

test('uses the actual catalog for source, documentation, Dockerfile and vendor paths', async t => {
  const cases = [
    ['src/Example.java', true], ['src/Example.scala', true],
    ['src/native.c', true], ['src/native.cpp', true], ['src/native.hpp', true],
    ['ci/build.py', true], ['ci/build.sh', true], ['ci/build.js', true],
    ['pom.xml', true], ['.env', true],
    ['.github/workflows/build.yml', false], ['.github/workflows/gpu-ci.yml', false],
    ['.github/workflows/nested/build.yaml', false], ['.github/workflows/sonar-check.yml', true],
    ['.github/scripts/sonar-check/select.js', true], ['.github/scripts/sonar-check/build.sh', true],
    ['.github/scripts/sonar-check/scan.py', true], ['.github/scripts/sonar-check/README.md', false],
    ['.github/scripts/gpu-ci/run.sh', true], ['.github/scripts/gpu-ci/image.py', true],
    ['.github/scripts/gpu-ci/cache.js', true],
    ['.github/workflows/action-helper/entrypoint.sh', true], ['.github/workflows/action-helper/python/utils.py', true],
    ['README.md', false], ['guide.rst', false], ['notes.txt', false],
    ['docs/example.java', false], ['.github/workflows/docs/example.yaml', false],
    ['.github/workflows/README.yml', false], ['module/ReadMe.html', false],
    ['Dockerfile', false], ['build/Dockerfile.devel', false],
    ['.github/workflows/action-helper/Dockerfile', false],
    ['.devcontainer/Dockerfile.js', false], ['application.dockerfile', false],
    ['thirdparty/Example.java', false], ['target/Example.java', false],
    ['src/main/cpp/CMakeLists.txt', false],
  ];
  for (const [filename, expected] of cases) {
    await t.test(filename, async () => {
      const f = fixture([modified(filename)]);
      await select(f.args);
      assert.equal(f.outputs['run-scan'], String(expected));
      assert.equal(f.outputs.reason, expected ? 'code-changes' : 'no-code-changes');
    });
  }
});

test('handles both rename paths, deletions, empty diffs and mixed changes', async t => {
  const cases = [
    ['code moved into docs', [{ filename: 'docs/Example.java', previous_filename: 'src/Example.java', status: 'renamed' }], true],
    ['docs moved into code', [{ filename: 'src/Example.java', previous_filename: 'docs/Example.java', status: 'renamed' }], true],
    ['Dockerfile rename', [{ filename: '.devcontainer/Dockerfile.sh', previous_filename: 'Dockerfile', status: 'renamed' }], false],
    ['deleted source', [{ filename: 'src/Example.java', status: 'removed' }], true],
    ['deleted Dockerfile', [{ filename: 'Dockerfile', status: 'removed' }], false],
    ['archived unrelated workflow', [{ filename: '.github/workflows/blossom-ci.yml.deprecated', previous_filename: '.github/workflows/blossom-ci.yml', status: 'renamed' }], false],
    ['deleted unrelated workflow', [{ filename: '.github/workflows/gpu-ci.yml', status: 'removed' }], false],
    ['deleted Sonar workflow', [{ filename: '.github/workflows/sonar-check.yml', status: 'removed' }], true],
    ['renamed Sonar workflow', [{ filename: '.github/workflows/sonar-check.yml.deprecated', previous_filename: '.github/workflows/sonar-check.yml', status: 'renamed' }], true],
    ['renamed into Sonar workflow', [{ filename: '.github/workflows/sonar-check.yml', previous_filename: '.github/workflows/build.yml', status: 'renamed' }], true],
    ['empty diff', [], false],
    ['docs and Dockerfile', [modified('README.md'), modified('Dockerfile')], false],
    ['mixed source and docs', [modified('README.md'), modified('src/Example.java')], true],
    ['mixed source and vendor', [modified('thirdparty/A.java'), modified('src/native.h')], true],
    ['mixed source and unrelated workflow', [modified('.github/workflows/gpu-ci.yml'), modified('src/Example.java')], true],
    ['workflow and its script', [modified('.github/workflows/gpu-ci.yml'), modified('.github/scripts/gpu-ci/run.sh')], true],
  ];
  for (const [name, files, expected] of cases) {
    await t.test(name, async () => {
      const f = fixture(files);
      await select(f.args);
      assert.equal(f.outputs['run-scan'], String(expected));
    });
  }
});

test('skips the Blossom retirement diff with documentation and unrelated CI changes', async () => {
  const f = fixture([
    modified('.github/AGENTS.md'),
    modified('.github/cpu-ci-suites.json'),
    modified('.github/scripts/gpu-ci/README.md'),
    {
      filename: '.github/workflows/blossom-ci.yml.deprecated',
      previous_filename: '.github/workflows/blossom-ci.yml',
      status: 'renamed',
    },
    modified('.github/workflows/gpu-ci.yml'),
    modified('CONTRIBUTING.md'),
    modified('ci/README.md'),
  ]);
  await select(f.args);
  assert.equal(f.outputs['run-scan'], 'false');
  assert.equal(f.outputs.reason, 'no-code-changes');
});

test('honors the title skip before reading even an oversized diff', async () => {
  const f = fixture();
  f.pr.title = '[skip ci] Update source';
  f.pr.changed_files = 3001;
  await select(f.args);
  assert.equal(f.outputs['run-scan'], 'false');
  assert.equal(f.outputs.reason, 'title-skip');
  assert.equal(f.outputs['title-skip'], 'true');
  assert.equal(f.calls.pr.length, 1);
  assert.deepEqual(f.calls.files, []);
});

test('reads all pages before deciding an otherwise documentation-only diff', async () => {
  const files = Array.from({ length: 100 }, (_, index) => modified(`docs/page-${index}.md`));
  files.push(modified('src/Example.java'));
  const f = fixture(files);
  await select(f.args);
  assert.equal(f.outputs['run-scan'], 'true');
  assert.deepEqual(f.calls.files.map(request => request.page), [1, 2]);
  assert(f.calls.files.every(request => request.per_page === 100));
});

test('rejects invalid counts and malformed, incomplete or duplicate file responses', async t => {
  const cases = [
    ['missing count', f => { delete f.pr.changed_files; }, /changed_files must be an integer/],
    ['string count', f => { f.pr.changed_files = '1'; }, /changed_files must be an integer/],
    ['negative count', f => { f.pr.changed_files = -1; }, /changed_files must be an integer/],
    ['oversized count', f => { f.pr.changed_files = 3001; }, /0 to 3000/],
    ['truncated page', f => { f.pr.changed_files = 2; }, /truncated/],
    ['count below response', f => { f.pr.changed_files = 0; }, /exceeds changed_files/],
  ];
  for (const [name, change, error] of cases) {
    await t.test(name, async () => {
      const f = fixture();
      change(f);
      await rejected(f, error);
    });
  }
  for (const [name, data, error] of [
    ['null page', null, /invalid PR file-list page/],
    ['object page', {}, /invalid PR file-list page/],
    ['oversized page', Array.from({ length: 101 }, (_, i) => modified(`src/${i}.java`)), /invalid PR file-list page/],
    ['missing filename', [{ status: 'modified' }], /invalid filename/],
    ['absolute filename', [modified('/src/Example.java')], /invalid filename/],
    ['unknown status', [{ filename: 'src/Example.java', status: 'unknown' }], /invalid filename, status/],
    ['rename without old path', [{ filename: 'README.md', status: 'renamed' }], /previous path/],
    ['explicit null old path', [{ ...modified('README.md'), previous_filename: null }], /previous path/],
  ]) {
    await t.test(name, async () => {
      const f = fixture();
      f.args.github.rest.pulls.listFiles = async () => ({ data });
      await rejected(f, error);
    });
  }
  await t.test('duplicate across pages', async () => {
    const files = Array.from({ length: 100 }, (_, index) => modified(`docs/page-${index}.md`));
    files.push(modified('docs/page-0.md'));
    const f = fixture(files);
    await rejected(f, /file list contains duplicates/);
    assert.deepEqual(f.calls.files.map(request => request.page), [1, 2]);
  });
});

test('rejects unrelated events and inconsistent copied-push identities before API reads', async t => {
  const cases = [
    ['PR event', f => { f.args.context.eventName = 'pull_request'; }],
    ['main push', f => { f.args.context.ref = 'refs/heads/main'; }],
    ['invalid PR number', f => { f.args.context.ref = 'refs/heads/pull-request/0'; }],
    ['deleted ref', f => { f.args.context.payload.deleted = true; }],
    ['missing deletion state', f => { delete f.args.context.payload.deleted; }],
    ['different pushed SHA', f => { f.args.context.payload.after = otherSha; }],
    ['short context SHA', f => { f.args.context.sha = '1234'; }],
    ['foreign repository', f => { f.args.context.payload.repository.full_name = 'contributor/cudf-spark-jni'; }],
  ];
  for (const [name, change] of cases) {
    await t.test(name, async () => {
      const f = fixture();
      change(f);
      await rejected(f, /non-deleted push of an exact commit/);
      assert.deepEqual(f.calls, { pr: [], files: [] });
    });
  }
});

test('rejects stale or unsupported PR metadata before selecting files', async t => {
  for (const [name, change, error] of [
    ['advanced head', pr => { pr.head.sha = otherSha; }, /expected head=.*observed state=open, head=/],
    ['closed PR', pr => { pr.state = 'closed'; }, /observed state=closed/],
    ['unsupported base', pr => { pr.base.ref = 'release/25.12'; }, /observed target "release\/25.12"/],
    ['foreign base repository', pr => { pr.base.repo.full_name = 'contributor/cudf-spark-jni'; }, /invalid metadata/],
    ['missing title', pr => { delete pr.title; }, /invalid metadata/],
    ['scanner substitution branch', pr => { pr.head.ref = 'feature-${env.VALUE}'; }, /property expansion/],
  ]) {
    await t.test(name, async () => {
      const f = fixture();
      change(f.pr);
      await rejected(f, error);
      assert.deepEqual(f.calls.files, []);
    });
  }
});

test('rechecks head, target, base commit, title and count after listing files', async t => {
  for (const [name, change, error] of [
    ['head', pr => { pr.head.sha = otherSha; }, /expected head=.*observed state=open, head=/],
    ['target', pr => { pr.base.ref = 'main'; }, /expected head=.*target=release\/26.12.*target=main/],
    ['base commit', pr => { pr.base.sha = otherSha; }, /diff or title changed/],
    ['title', pr => { pr.title = '[skip ci] Changed during selection'; }, /diff or title changed/],
    ['count', pr => { pr.changed_files = 2; }, /diff or title changed/],
  ]) {
    await t.test(name, async () => {
      const f = fixture();
      const before = structuredClone(f.pr);
      const after = structuredClone(f.pr);
      change(after);
      let reads = 0;
      f.args.github.rest.pulls.get = async () => ({ data: ++reads === 1 ? before : after });
      await rejected(f, error);
      assert.equal(reads, 2);
      assert.equal(f.calls.files.length, 1);
    });
  }
});

test('preserves API failures without reporting a successful skip or retrying', async t => {
  for (const method of ['get', 'listFiles']) {
    await t.test(method, async () => {
      const f = fixture();
      const error = Object.assign(new Error(`GitHub ${method} returned HTTP 503`), { status: 503 });
      let requests = 0;
      f.args.github.rest.pulls[method] = async () => { requests++; throw error; };
      await rejected(f, actual => actual === error);
      assert.equal(requests, 1);
    });
  }
});

test('noninitial checks bind the selected head, target and PR without relisting files', async t => {
  for (const base of ['main', 'release/26.12']) {
    await t.test(base, async () => {
      const f = fixture();
      f.args.initial = false;
      f.pr.base.ref = base;
      f.args.env.APPROVED_BASE_REF = base;
      await select(f.args);
      assert.deepEqual(f.outputs, {});
      assert.equal(f.calls.pr.length, 1);
      assert.deepEqual(f.calls.files, []);
    });
  }
  for (const [name, change, error] of [
    ['advanced head', f => { f.pr.head.sha = otherSha; }, /expected head=.*observed state=open, head=/],
    ['retargeted PR', f => { f.pr.base.ref = 'main'; }, /expected head=.*target=release\/26.12.*target=main/],
    ['new title skip', f => { f.pr.title = '[skip ci] Do not scan'; }, /now requests CI to be skipped/],
    ['different selected SHA', f => { f.args.env.APPROVED_SHA = otherSha; }, /does not match the selected PR revision/],
    ['different selected PR', f => { f.args.env.PR_NUMBER = '8'; }, /does not match the selected PR revision/],
    ['missing selected base', f => { delete f.args.env.APPROVED_BASE_REF; }, /observed target undefined/],
  ]) {
    await t.test(name, async () => {
      const f = fixture();
      f.args.initial = false;
      change(f);
      await rejected(f, error);
      assert.deepEqual(f.calls.files, []);
    });
  }
});
