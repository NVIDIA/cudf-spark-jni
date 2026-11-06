#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect this pilot's completed analysis for the hosted GitHub reporter."""

import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ValidationError(ValueError):
    """A diagnostic constructed locally, without server responses or credentials."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise ValidationError("Sonar API redirected; refusing to forward credentials")


class Sonar:
    def __init__(self, host, token):
        self.host = host.rstrip("/")
        url = urllib.parse.urlsplit(self.host)
        require(url.scheme == "https" and url.netloc and not url.username
                and not url.query and not url.fragment, "Invalid Sonar service URL")
        self.authorization = "Basic " + base64.b64encode((token + ":").encode()).decode()
        self.opener = urllib.request.build_opener(NoRedirect())
        self.deadline = time.monotonic() + 600

    def get(self, endpoint, **params):
        url = self.host + "/api/" + endpoint + "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"Authorization": self.authorization})
        for attempt in range(3):
            remaining = self.deadline - time.monotonic()
            require(remaining > 0, "Sonar result collection timed out")
            try:
                with self.opener.open(request, timeout=min(30, remaining)) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code not in (429, 502, 503, 504) or attempt == 2:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(min(3, max(0, self.deadline - time.monotonic())))
        raise RuntimeError("Sonar request did not complete")


def collect(sonar, result, task_file, scanner_outcome):
    project, pr = result["project"], str(result["pr"])
    properties = dict(line.split("=", 1) for line in task_file.read_text().splitlines()
                      if "=" in line)
    require(properties.get("projectKey") == project, "Unexpected report project")
    require(properties.get("serverUrl", "").rstrip("/") == sonar.host,
            "Unexpected report server")
    dashboard = urllib.parse.urlsplit(properties.get("dashboardUrl", ""))
    expected_host = urllib.parse.urlsplit(sonar.host)
    query = urllib.parse.parse_qs(dashboard.query)
    require((dashboard.scheme, dashboard.netloc) == (expected_host.scheme, expected_host.netloc)
            and query.get("id") == [project] and query.get("pullRequest") == [pr],
            "Unexpected report pull request")
    task_id = properties.get("ceTaskId", "")
    require(re.fullmatch(r"[A-Za-z0-9_-]+", task_id), "Missing or invalid CE task ID")
    while True:
        task = sonar.get("ce/task", id=task_id)["task"]
        require(task.get("id") == task_id and task.get("componentKey") == project
                and str(task.get("pullRequest")) == pr, "Unexpected CE task identity")
        status = task["status"]
        if status == "SUCCESS":
            break
        require(status in ("PENDING", "IN_PROGRESS"), "Sonar analysis did not complete")
        require(time.monotonic() < sonar.deadline, "Sonar analysis timed out")
        time.sleep(3)
    analysis_id = task.get("analysisId")
    require(isinstance(analysis_id, str) and analysis_id, "CE task has no analysis ID")
    gate = sonar.get("qualitygates/project_status", analysisId=analysis_id)["projectStatus"]
    require(gate.get("status") in ("OK", "ERROR", "WARN", "NONE"),
            "Unknown quality gate result")
    conditions = {item["metricKey"]: item for item in gate.get("conditions", [])}
    counts = {}
    for severity in ("high", "blocker"):
        condition = conditions.get("new_software_quality_" + severity + "_issues", {})
        require(condition.get("comparator") == "GT"
                and condition.get("errorThreshold") == "0",
                "Expected High/Blocker quality gate condition is missing or changed")
        value = condition.get("actualValue", "")
        require(isinstance(value, str) and value.isdigit(), "Missing quality gate issue count")
        counts[severity] = int(value)

    # This endpoint tracks the whole project, not just this PR. Conservatively reject
    # any intervening analysis rather than attach another scan's issues to this SHA.
    def check_current():
        current = sonar.get("ce/component", component=project).get("current", {})
        require(current.get("id") == task_id and current.get("analysisId") == analysis_id
                and current.get("status") == "SUCCESS" and str(current.get("pullRequest")) == pr,
                "Another project analysis replaced this result; reapprove and rerun")

    check_current()
    issues, seen, total = [], set(), None
    for page in range(1, 21):
        response = sonar.get("issues/search", components=project, pullRequest=pr,
                             inNewCodePeriod="true", issueStatuses="OPEN,CONFIRMED",
                             p=page, ps=500)
        count = response["paging"]["total"]
        require(isinstance(count, int) and 0 <= count <= 10000, "Unsupported issue count")
        require(total is None or count == total, "Issue count changed during collection")
        total = count
        rows = response.get("issues", [])
        for issue in rows:
            require(issue.get("project") == project and str(issue.get("pullRequest")) == pr,
                    "Unexpected issue identity")
            key = issue["key"]
            require(key not in seen, "Duplicate issue during pagination")
            seen.add(key)
            component = issue["component"]
            require(component.startswith(project + ":"), "Unexpected issue component")
            issues.append({"key": key, "rule": issue["rule"], "message": issue["message"],
                           "path": component[len(project) + 1:], "line": issue.get("line"),
                           "impacts": issue.get("impacts", []),
                           "legacySeverity": issue.get("severity"), "legacyType": issue.get("type")})
        if len(seen) == total:
            break
        require(rows and len(seen) < total, "Incomplete issue pagination")
    require(len(seen) == total, "Incomplete issue pagination")
    check_current()
    severity_order = {"BLOCKER": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    issues.sort(key=lambda issue: min((severity_order.get(impact["severity"], 5)
                                      for impact in issue["impacts"]), default=5))
    result.update(complete=True, taskId=task_id, analysisId=analysis_id,
                  gateStatus=gate["status"], high=counts["high"], blocker=counts["blocker"],
                  totalIssues=total, issues=issues[:10],
                  reportUrl=sonar.host + "/dashboard?" + urllib.parse.urlencode(
                      {"id": project, "pullRequest": pr}))
    result["passed"] = (scanner_outcome == "success" and gate["status"] == "OK"
                        and counts["high"] == counts["blocker"] == 0
                        and not any(impact["severity"] in ("HIGH", "BLOCKER")
                                    for issue in issues for impact in issue["impacts"]))


def main():
    result = {"schema": 1, "sha": os.environ["APPROVED_SHA"],
              "pr": int(os.environ["PR_NUMBER"]), "project": os.environ["SONAR_PROJECT_KEY"],
              "runId": os.environ["GITHUB_RUN_ID"],
              "runAttempt": os.environ["GITHUB_RUN_ATTEMPT"],
              "complete": False, "passed": False}
    try:
        actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        require(actual_sha == result["sha"], "Checkout no longer matches the approved commit")
        sonar = Sonar(os.environ["SONAR_HOST_URL"], os.environ["SONAR_TOKEN"])
        collect(sonar, result, Path(".scannerwork/report-task.txt"), os.environ["SCANNER_OUTCOME"])
    except Exception as error:
        # Publish no credential-bearing response bodies or raw API errors in the artifact.
        detail = str(error) if isinstance(error, ValidationError) else type(error).__name__
        if isinstance(error, urllib.error.HTTPError):
            detail = f"HTTP {error.code}"
        print(f"Sonar result collection failed: {detail}.", file=sys.stderr)
        result["error"] = "Scan result collection failed; see the workflow logs."
    Path(sys.argv[1]).write_text(json.dumps(result) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
