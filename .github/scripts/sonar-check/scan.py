#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scan source and class files as data; never load PR scripts or scanner settings."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def cleanup():
    """Remove only this invocation's scratch directories, never the runner HOME."""
    temporary = Path(os.environ["RUNNER_TEMP"]).resolve()
    for name in ("SONAR_WORK_DIRECTORY", "CLASSES_DIRECTORY"):
        value = os.environ.get(name)
        if not value:
            continue
        path = Path(value)
        if (
            path.is_symlink()
            or path.resolve().parent != temporary
            or not re.fullmatch(r"sonar-[A-Za-z0-9_-]+", path.name)
        ):
            raise ValueError("Refusing cleanup outside this scan's temporary directories")
        if path.exists():
            shutil.rmtree(path)


def validate_classes(classes):
    """Bound untrusted artifact contents before transferring them internally."""
    if classes.is_symlink() or not classes.is_dir():
        raise ValueError("The compile artifact is not a regular class directory")
    count = total = main_classes = 0
    for path in classes.rglob("*"):
        if path.is_symlink():
            raise ValueError("The compile artifact must not contain symlinks")
        relative = path.relative_to(classes)
        if relative.parts[0] not in ("classes", "test-classes"):
            raise ValueError("The compile artifact contains an unexpected directory")
        if path.is_dir():
            continue
        if not path.is_file() or path.suffix != ".class":
            raise ValueError("The compile artifact must contain only class files")
        count += 1
        total += path.stat().st_size
        main_classes += relative.parts[0] == "classes"
        if count > 50000 or total > 512 * 1024 * 1024:
            raise ValueError("Class artifact exceeds 50000 files or 512 MiB")
    if not main_classes:
        raise ValueError("The producing compile job supplied no Java class files")


def main():
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    source = workspace / "source"
    classes = Path(os.environ["CLASSES_DIRECTORY"])
    work = Path(tempfile.mkdtemp(prefix="sonar-", dir=os.environ["RUNNER_TEMP"]))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"task-file={work / 'report-task.txt'}\n")
        output.write(f"result-path={work / 'result.json'}\n")
        output.write(f"work-directory={work}\n")

    expected = os.environ["APPROVED_SHA"]
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", expected) or actual != expected:
        raise ValueError("Source checkout no longer matches the approved commit")
    if os.environ.get("GITHUB_RUNNER_ADMIN_TOKEN"):
        raise ValueError("Runner administration credentials must not enter the scan")
    if not os.environ.get("SONAR_TOKEN") or not os.environ.get("SONAR_HOST_URL"):
        raise ValueError("Configure the existing SONAR_TOKEN and SONAR_HOST_URL secrets")
    validate_classes(classes)
    classes = classes.resolve()

    # Read version metadata without invoking Maven or any code from the PR.
    pom = ET.parse(source / "pom.xml")
    version = pom.findtext("{http://maven.apache.org/POM/4.0.0}version", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", version):
        raise ValueError("The checked-out POM has no supported project version")

    # Explicit settings suppress PR-owned sonar-project.properties. The scanner
    # uses cwd as its project root; project.settings cannot be combined with
    # sonar.projectBaseDir. Working files and the cache are fresh for this scan.
    settings = work / "sonar.properties"
    settings.write_text("# Configuration is owned by the trusted workflow.\n")
    catalog = json.loads((Path(__file__).resolve().parents[2] / "sonar-paths.json").read_text())
    inclusions = [f"**/*.{suffix}" for suffix in catalog["extensions"]]
    inclusions.extend(catalog["paths"])
    exclusions = [value + "**" for value in catalog["excluded-paths"]]
    properties = {
        "project.settings": settings,
        "sonar.ci.autoconfig.disabled": "true",
        "sonar.projectKey": os.environ["SONAR_PROJECT_KEY"],
        "sonar.projectVersion": version,
        "sonar.scm.revision": expected,
        "sonar.pullrequest.key": os.environ["PR_NUMBER"],
        "sonar.pullrequest.branch": os.environ["PR_BRANCH"],
        "sonar.pullrequest.base": os.environ["PR_BASE_REF"],
        "sonar.java.binaries": classes / "classes",
        "sonar.sources": ".",
        "sonar.inclusions": ",".join(inclusions),
        "sonar.exclusions": ",".join([*exclusions, "src/test/java/**"]),
        "sonar.docker.file.patterns": "**/Dockerfile*,**/dockerfile*,**/*.dockerfile",
        "sonar.scanner.excludeHiddenFiles": "false",
        "sonar.working.directory": work / "scannerwork",
        "sonar.scanner.metadataFilePath": work / "report-task.txt",
    }
    if (source / "src/test/java").is_dir():
        properties["sonar.tests"] = "src/test/java"
        properties["sonar.test.inclusions"] = "**/*.java"
        properties["sonar.java.test.binaries"] = classes / "test-classes"
    # Scanner properties expand ${env.NAME}, even in quoted -D arguments. Never
    # let a PR branch name interpolate a credential into submitted metadata.
    if any("${" in str(value) for value in properties.values()):
        raise ValueError("Sonar metadata must not contain scanner substitution syntax")
    # The collector owns server-task waiting and the quality gate. A gate failure
    # is a verified analysis result, rather than a scanner process failure.
    subprocess.run(
        ["sonar-scanner", *(f"-D{key}={value}" for key, value in properties.items())],
        cwd=source,
        env={**os.environ, "SONAR_USER_HOME": str(work / "cache")},
        check=True,
        timeout=900,
    )


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--cleanup"]:
            cleanup()
        elif sys.argv[1:] == ["--check-classes"]:
            validate_classes(Path(os.environ["CLASSES_DIRECTORY"]))
        elif sys.argv[1:]:
            raise ValueError("Unsupported Sonar scan arguments")
        else:
            main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Sonar scanner command failed with exit {error.returncode}; see scanner logs") from None
    except subprocess.TimeoutExpired:
        raise SystemExit("Sonar scanner exceeded its 15-minute submission budget") from None
    except (OSError, ValueError, ET.ParseError) as error:
        raise SystemExit(f"Sonar scan preparation failed: {error}") from None
