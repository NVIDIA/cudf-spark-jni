#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""Create immutable-per-build JNI candidate identities and build manifests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional
import xml.etree.ElementTree as ET


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CANDIDATE_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)-jni\.(?P<sha>[0-9a-f]{12})\.r(?P<revision>[1-9]\d*)-SNAPSHOT$"
)
MAVEN_NAMESPACE = {"m": "http://maven.apache.org/POM/4.0.0"}


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def pom_version(pom: Path) -> str:
    root = ET.parse(pom).getroot()
    version = root.find("m:version", MAVEN_NAMESPACE)
    if version is None or not version.text:
        raise ValueError(f"project version is missing from {pom}")
    return version.text.strip()


def base_version(version: str) -> str:
    match = re.fullmatch(r"(\d+\.\d+\.\d+)-SNAPSHOT", version)
    if not match:
        candidate = CANDIDATE_RE.fullmatch(version)
        if candidate:
            return candidate.group("base")
        raise ValueError(
            "project version must be <major>.<minor>.<patch>-SNAPSHOT or a JNI candidate"
        )
    return match.group(1)


def validate_sha(value: str, label: str) -> str:
    normalized = value.lower()
    if not SHA_RE.fullmatch(normalized):
        raise ValueError(f"{label} must be a full 40-character hexadecimal commit SHA")
    return normalized


def candidate_version(version: str, jni_sha: str, revision: int) -> str:
    if revision < 1:
        raise ValueError("artifact revision must be greater than zero")
    sha = validate_sha(jni_sha, "JNI SHA")
    return f"{base_version(version)}-jni.{sha[:12]}.r{revision}-SNAPSHOT"


def prepare_pom(pom: Path, jni_sha: str, revision: int) -> str:
    current = pom_version(pom)
    candidate = candidate_version(current, jni_sha, revision)
    text = pom.read_text()
    marker = f"<version>{current}</version>"
    if text.count(marker) != 1:
        raise ValueError(f"expected exactly one project version marker in {pom}")
    pom.write_text(text.replace(marker, f"<version>{candidate}</version>", 1))
    return candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(
    path: Path, published_name: str, extension: str, classifier: Optional[str]
) -> dict:
    return {
        "publishedName": published_name,
        "extension": extension,
        "classifier": classifier,
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def create_manifest(args: argparse.Namespace) -> dict:
    pom = args.pom.resolve()
    repo = args.repo.resolve()
    version = pom_version(pom)
    if not CANDIDATE_RE.fullmatch(version):
        raise ValueError(f"POM version is not a JNI candidate: {version}")

    jni_sha = validate_sha(args.jni_sha or run_git(repo, "rev-parse", "HEAD"), "JNI SHA")
    expected = candidate_version(version, jni_sha, args.revision)
    if version != expected:
        raise ValueError(f"POM candidate {version} does not match expected {expected}")

    cudf_sha = validate_sha(
        args.cudf_sha or run_git(repo / "thirdparty" / "cudf", "rev-parse", "HEAD"),
        "cuDF SHA",
    )
    required = sorted(set(args.classifier))
    default_classifier = args.default_classifier or required[0]
    if default_classifier not in required:
        raise ValueError("default classifier must also be a required classifier")
    artifact_dir = args.artifact_dir.resolve()
    artifact_id = args.artifact_id
    missing = []
    files = []
    for classifier in required:
        path = artifact_dir / f"{artifact_id}-{version}-{classifier}.jar"
        if not path.is_file():
            missing.append(str(path))
        else:
            files.append((path, path.name, "jar", classifier))

    for suffix in ("sources.jar", "javadoc.jar"):
        path = artifact_dir / f"{artifact_id}-{version}-{suffix}"
        if not path.is_file():
            missing.append(str(path))
        else:
            classifier = suffix.removesuffix(".jar")
            files.append((path, path.name, "jar", classifier))

    if missing:
        raise ValueError("required candidate artifacts are missing:\n" + "\n".join(missing))

    default_path = artifact_dir / f"{artifact_id}-{version}-{default_classifier}.jar"
    files.append((default_path, f"{artifact_id}-{version}.jar", "jar", None))
    files.append((pom, f"{artifact_id}-{version}.pom", "pom", None))

    manifest = {
        "schemaVersion": 1,
        "candidateId": version,
        "artifactRevision": args.revision,
        "source": {
            "jniCommit": jni_sha,
            "cudfCommit": cudf_sha,
        },
        "build": {
            "buildTag": os.environ.get("BUILD_TAG", ""),
            "buildUrl": os.environ.get("BUILD_URL", ""),
        },
        "requiredClassifiers": required,
        "defaultClassifier": default_classifier,
        "artifacts": [
            artifact_entry(path, published_name, extension, classifier)
            for path, published_name, extension, classifier in sorted(
                files, key=lambda entry: entry[1]
            )
        ],
    }
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version")
    version_parser.add_argument("--pom", type=Path, default=Path("pom.xml"))
    version_parser.add_argument("--repo", type=Path, default=Path("."))
    version_parser.add_argument("--jni-sha")
    version_parser.add_argument("--revision", type=int, default=1)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--pom", type=Path, default=Path("pom.xml"))
    prepare_parser.add_argument("--repo", type=Path, default=Path("."))
    prepare_parser.add_argument("--jni-sha")
    prepare_parser.add_argument("--revision", type=int, default=1)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--pom", type=Path, default=Path("pom.xml"))
    manifest_parser.add_argument("--repo", type=Path, default=Path("."))
    manifest_parser.add_argument("--artifact-dir", type=Path, default=Path("target"))
    manifest_parser.add_argument("--artifact-id", default="cudf-spark-jni")
    manifest_parser.add_argument("--classifier", action="append", required=True)
    manifest_parser.add_argument("--default-classifier")
    manifest_parser.add_argument("--jni-sha")
    manifest_parser.add_argument("--cudf-sha")
    manifest_parser.add_argument("--revision", type=int, default=1)
    manifest_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.command == "version":
            sha = args.jni_sha or run_git(args.repo.resolve(), "rev-parse", "HEAD")
            print(candidate_version(pom_version(args.pom), sha, args.revision))
        elif args.command == "prepare":
            sha = args.jni_sha or run_git(args.repo.resolve(), "rev-parse", "HEAD")
            print(prepare_pom(args.pom, sha, args.revision))
        else:
            manifest = create_manifest(args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
