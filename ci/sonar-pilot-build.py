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

"""Compile the approved pilot source without Sonar credentials."""

import os
from pathlib import Path
import re
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET


def main():
    approved = os.environ["APPROVED_SHA"]
    if not re.fullmatch(r"[a-fA-F0-9]{40}", approved):
        raise ValueError("The approved SHA must be a full commit SHA")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if actual.lower() != approved.lower():
        raise ValueError("Checkout differs from the approved commit")

    mirror = os.environ["MAVEN_MIRROR"].strip()
    try:
        url = urllib.parse.urlsplit(mirror)
        port = url.port
    except ValueError:
        raise ValueError("The Maven mirror must be a valid HTTPS URL") from None
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or "\\" in mirror
        or any(character.isspace() for character in mirror)
        or any(ord(character) < 32 for character in mirror)
    ):
        raise ValueError("The Maven mirror must use HTTPS without credentials or query")

    settings_path = Path(os.environ["MAVEN_SETTINGS"]).resolve()
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    if settings_path == runner_temp or not settings_path.is_relative_to(runner_temp):
        raise ValueError("Maven settings must be a file inside RUNNER_TEMP")
    settings = ET.Element("settings", xmlns="http://maven.apache.org/SETTINGS/1.0.0")
    entry = ET.SubElement(ET.SubElement(settings, "mirrors"), "mirror")
    for key, value in (
        ("id", "sonar-pilot"),
        ("mirrorOf", "*"),
        ("url", mirror),
    ):
        ET.SubElement(entry, key).text = value

    # Own only this new settings file; do not overwrite another step's settings.
    descriptor = os.open(settings_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            ET.ElementTree(settings).write(output, encoding="utf-8", xml_declaration=True)
        maven = ["mvn", "-s", str(settings_path), "-B"]
        subprocess.run(
            [
                *maven,
                "-Dmaven.wagon.http.retryHandler.count=3",
                "compile",
                "-DskipTests=true",
                "-Dmaven.antrun.skip=true",
            ],
            check=True,
            timeout=1800,
        )
        version = subprocess.check_output(
            [
                *maven,
                "-Dstyle.color=never",
                "help:evaluate",
                "-Dexpression=project.version",
                "-q",
                "-DforceStdout",
            ],
            text=True,
            timeout=180,
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", version):
            raise ValueError("Maven returned an invalid project version")
        if not Path("target/classes").is_dir():
            raise ValueError("Java compilation did not produce target/classes")
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"version={version}\n")
    finally:
        settings_path.unlink()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        raise SystemExit("Pilot Java compilation failed: " + message) from None
