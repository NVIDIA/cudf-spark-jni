// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the Apache License, Version 2.0.

module.exports = async function approve({github, context, core, initial = false}) {
  const {owner, repo} = context.repo;
  const eventPR = context.payload.pull_request;
  const {data: permission} = await github.rest.repos.getCollaboratorPermissionLevel({
    owner,
    repo,
    username: process.env.GITHUB_TRIGGERING_ACTOR,
  });
  if (!['admin', 'maintain', 'write'].includes(permission.permission)) {
    throw new Error('A repository maintainer must approve this pilot.');
  }
  const {data: pr} = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: eventPR.number,
  });
  const sha = process.env.APPROVED_SHA || eventPR.head.sha;
  const base = process.env.APPROVED_BASE || eventPR.base.ref;
  if (pr.number !== 5433 || pr.state !== 'open' ||
      pr.head.repo?.full_name !== `${owner}/${repo}` ||
      pr.head.sha !== sha || pr.base.ref !== base ||
      !pr.labels.some(label => label.name === 'sonar-runner-pilot')) {
    throw new Error('PR approval changed; review the current head and reapply the label.');
  }
  if (initial) {
    if (!process.env.SONAR_PROJECT_KEY?.trim() || !process.env.MAVEN_MIRROR?.trim()) {
      throw new Error('Configure the Sonar project and anonymous Maven mirror.');
    }
    core.setOutput('sha', sha);
    core.setOutput('branch', pr.head.ref);
    core.setOutput('base', base);
  }
};
