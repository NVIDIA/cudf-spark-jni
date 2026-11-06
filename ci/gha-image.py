#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the approved main-targeting PR's public toolchain, without Docker credentials."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def get(url, token=None, allow_missing=False):
    headers = {"Accept": ACCEPT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            return json.load(response), response.headers
    except HTTPError as error:
        if allow_missing and error.code == 404:
            return None, None
        raise


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Registry returned an invalid image digest")
    return value


def resolve(selected):
    image = selected["image"]
    if not image.startswith("ghcr.io/"):
        raise ValueError("JNI toolchain must use GHCR")
    path = image.removeprefix("ghcr.io/")
    token_data, _ = get(f"https://ghcr.io/token?service=ghcr.io&scope=repository:{path}:pull")
    token = token_data["token"]
    for tag in selected["tags"]:
        manifest, headers = get(f"https://ghcr.io/v2/{path}/manifests/{tag}", token, True)
        if manifest is None:
            print(f"No published toolchain: {image}:{tag}")
            continue
        image_digest = headers.get("Docker-Content-Digest")
        if "manifests" in manifest:
            matches = [entry for entry in manifest["manifests"]
                       if entry.get("platform", {}).get("os") == "linux" and
                       entry.get("platform", {}).get("architecture") == "amd64"]
            if not matches:
                print(f"No linux/amd64 toolchain: {image}:{tag}")
                continue
            if len(matches) != 1:
                raise ValueError("Ambiguous linux/amd64 image manifest")
            image_digest = digest(matches[0]["digest"])
            manifest, _ = get(f"https://ghcr.io/v2/{path}/manifests/{image_digest}", token)
        image_digest = digest(image_digest)
        config_digest = digest(manifest["config"]["digest"])
        config, _ = get(f"https://ghcr.io/v2/{path}/blobs/{config_digest}", token)
        if (config.get("os") == "linux" and config.get("architecture") == "amd64" and
                (config.get("config", {}).get("Labels") or {}).get(
                    "org.opencontainers.image.version") == selected["version"]):
            reference = f"{image}@{image_digest}"
            print(f"Compatible toolchain {image}:{tag} -> {reference}")
            return reference
        print(f"Published toolchain does not match the PR recipe: {image}:{tag}")
    print("Build the missing/incompatible toolchain on this worker; do not publish PR images")
    return ""


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("cuda12", "cuda13"):
        sys.exit("Usage: gha-image.py cuda12|cuda13")
    classifier = sys.argv[1]
    selected = json.loads(subprocess.check_output([
        sys.executable, "ci/image-config.py", "select", "--classifiers", classifier,
        "--allow-main-fallback",
    ], text=True))[classifier]
    reference = resolve(selected)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"reference={reference}\nrecipe={selected['version']}\n")
        output.write("config=" + json.dumps(selected["config"], separators=(",", ":")) + "\n")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        sys.exit(f"GHCR request failed with HTTP {error.code}; not a missing-image fallback. "
                 "Check registry access and that the JNI image package is published and public.")
