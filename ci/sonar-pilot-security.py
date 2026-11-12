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

"""Pre-merge pilot validation of an unpublished scanner image; never publish it."""

import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile

COMMON_SHA = "962872d1f2f2647df7da157941b8496c3bd25a9f"


def main():
    if (
        os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
        or platform.system() != "Linux"
        or platform.machine().lower() not in ("x86_64", "amd64")
    ):
        raise ValueError("Pilot image validation requires a hosted Linux AMD64 runner")
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    common = (workspace / "common").resolve()
    source = (workspace / "source").resolve()
    if not common.is_relative_to(workspace) or not source.is_relative_to(workspace):
        raise ValueError("Pilot checkouts must remain inside GITHUB_WORKSPACE")
    git_environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "LANG": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    git = ["/usr/bin/git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null"]
    actual = subprocess.check_output(
        [*git, "rev-parse", "HEAD"],
        cwd=common,
        env=git_environment,
        text=True,
        timeout=30,
    ).strip()
    if actual != COMMON_SHA:
        raise ValueError("Common checkout differs from the reviewed pilot revision")
    subprocess.run(
        [*git, "diff", "--quiet", "HEAD", "--"],
        cwd=common,
        env=git_environment,
        check=True,
        timeout=30,
    )

    spec = importlib.util.spec_from_file_location(
        "pilot_script_scan", common / "sonar-check/script_scan.py"
    )
    scanner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scanner
    spec.loader.exec_module(scanner)
    previous_handler = signal.signal(signal.SIGTERM, scanner.cancelled)
    try:
        approved = scanner.checkout_sha(source, os.environ["APPROVED_SHA"])
        catalog = json.loads((common / ".github/image-builds.json").read_text())
        recipe = catalog["sonar-script-scan"]
        with tempfile.TemporaryDirectory(
            prefix="sonar-pilot-security-", dir=os.environ["RUNNER_TEMP"]
        ) as directory:
            temporary = Path(directory)
            build_directory = temporary / "build"
            build_directory.mkdir()
            image_id_file = temporary / "image-id"
            docker = scanner.docker_command(build_directory)
            # This is the existing image smoke-test path, not a production fallback.
            # The public action still requires an anonymously pullable published image.
            scanner.command_output(
                [
                    *docker,
                    "build",
                    "--pull",
                    "--iidfile",
                    str(image_id_file),
                    "--file",
                    str(common / recipe["dockerfile"]),
                    "--platform",
                    "linux/amd64",
                    str(common / recipe["context"]),
                ],
                limit=1024 * 1024,
                timeout=300,
                phase="Pilot scanner image build",
            )
            image = image_id_file.read_text().strip()
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
                raise ValueError("Image build did not return an immutable local image ID")
            scan_directory = temporary / "scan"
            scan_directory.mkdir()
            result = scanner.run(
                source, approved, scan_directory, image, platform="linux/amd64"
            )
        report = (
            "Pre-merge pilot: validated the reviewed image recipe locally. "
            "This does not validate production image publication or anonymous pulling.\n\n"
            + scanner.summary(result)
        )
        print(report)
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(report + "\n")
        if result["findings"]:
            return 1
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write(f"passed=true\nsha={approved}\n")
        return 0
    except scanner.ScanError as error:
        print("::error::" + str(error))
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Pilot script security scan was cancelled") from None
    except (ValueError, OSError, KeyError, subprocess.SubprocessError):
        raise SystemExit("Pilot script security validation could not complete") from None
