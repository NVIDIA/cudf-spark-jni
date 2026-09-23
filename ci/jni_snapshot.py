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

"""Create and validate provenance for a moving JNI SNAPSHOT artifact bundle."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, Optional, Tuple
import zipfile


JNI_PROPERTIES = "cudf-spark-jni-version-info.properties"
CUDF_PROPERTIES = "cudf-java-version-info.properties"
SHA_RE = re.compile(r"[0-9a-f]{40}")
MOVING_SNAPSHOT_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+-SNAPSHOT")


def validate_sha(value: str, label: str) -> str:
    value = value.lower()
    if not SHA_RE.fullmatch(value):
        raise ValueError(f"{label} must be a full 40-character hexadecimal commit SHA")
    return value


def validate_version(value: str) -> str:
    if not MOVING_SNAPSHOT_RE.fullmatch(value):
        raise ValueError(f"logical version must be a release-train moving SNAPSHOT, got {value}")
    return value


def parse_key_value(value: str) -> Tuple[str, str]:
    key, separator, item = value.partition("=")
    if not separator or not key or not item:
        raise argparse.ArgumentTypeError("build input must be KEY=VALUE")
    return key, item


def parse_artifact(value: str) -> Tuple[Optional[str], str, Path]:
    classifier, separator, remainder = value.partition(":")
    extension, separator2, path = remainder.partition(":")
    if not separator or not separator2 or not extension or not path:
        raise argparse.ArgumentTypeError(
            "artifact must be CLASSIFIER:EXTENSION:PATH; use '-' for no classifier"
        )
    return (None if classifier == "-" else classifier, extension, Path(path))


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def jar_properties(path: Path, member: str) -> Dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        content = archive.read(member).decode()
    properties = {}
    for line in content.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key.strip()] = value.strip()
    return properties


def verify_runtime_jar(path: Path, jni_sha: str, cudf_sha: str) -> None:
    actual_jni = jar_properties(path, JNI_PROPERTIES).get("revision", "").lower()
    actual_cudf = jar_properties(path, CUDF_PROPERTIES).get("revision", "").lower()
    if actual_jni != jni_sha:
        raise ValueError(f"{path} embeds JNI revision {actual_jni}, expected {jni_sha}")
    if actual_cudf != cudf_sha:
        raise ValueError(f"{path} embeds cuDF revision {actual_cudf}, expected {cudf_sha}")


def canonical_fingerprint(source: dict, logical_version: str, inputs: dict) -> str:
    content = json.dumps(
        {"source": source, "logicalVersion": logical_version, "inputs": inputs},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(content).hexdigest()


def artifact_record(
    classifier: Optional[str], extension: str, path: Path, jni_sha: str, cudf_sha: str
) -> dict:
    if not path.is_file():
        raise ValueError(f"artifact does not exist: {path}")
    if extension == "jar" and classifier and classifier.startswith("cuda"):
        verify_runtime_jar(path, jni_sha, cudf_sha)
    return {
        "classifier": classifier,
        "extension": extension,
        "fileName": path.name,
        "sha1": digest(path, "sha1"),
        "sha256": digest(path, "sha256"),
        "size": path.stat().st_size,
    }


def create_manifest(
    jni_sha: str,
    cudf_sha: str,
    logical_version: str,
    artifacts: Iterable[Tuple[Optional[str], str, Path]],
    build_inputs: Iterable[Tuple[str, str]],
) -> dict:
    jni_sha = validate_sha(jni_sha, "JNI SHA")
    cudf_sha = validate_sha(cudf_sha, "cuDF SHA")
    logical_version = validate_version(logical_version)
    inputs = dict(build_inputs)
    source = {"jniSha": jni_sha, "cudfSha": cudf_sha}
    records = [
        artifact_record(classifier, extension, path, jni_sha, cudf_sha)
        for classifier, extension, path in artifacts
    ]
    if not records:
        raise ValueError("at least one artifact is required")
    identities = [(record["extension"], record["classifier"]) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("artifact classifier and extension pairs must be unique")
    records.sort(key=lambda record: (record["extension"], record["classifier"] or ""))
    return {
        "schemaVersion": 1,
        "logicalVersion": logical_version,
        "source": source,
        "build": {
            "fingerprint": canonical_fingerprint(source, logical_version, inputs),
            "inputs": dict(sorted(inputs.items())),
        },
        "artifacts": records,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }


def verify_manifest(manifest: dict, root: Path) -> None:
    validate_version(manifest["logicalVersion"])
    jni_sha = validate_sha(manifest["source"]["jniSha"], "JNI SHA")
    cudf_sha = validate_sha(manifest["source"]["cudfSha"], "cuDF SHA")
    expected_fingerprint = canonical_fingerprint(
        manifest["source"], manifest["logicalVersion"], manifest["build"]["inputs"]
    )
    if manifest["build"].get("fingerprint") != expected_fingerprint:
        raise ValueError("build fingerprint does not match the manifest inputs")
    for record in manifest["artifacts"]:
        path = root / record["fileName"]
        actual = artifact_record(
            record.get("classifier"), record["extension"], path, jni_sha, cudf_sha
        )
        for field in ("sha1", "sha256", "size"):
            if actual[field] != record[field]:
                raise ValueError(f"artifact {field} mismatch for {path}")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--jni-sha", required=True)
    manifest.add_argument("--cudf-sha", required=True)
    manifest.add_argument("--logical-version", required=True)
    manifest.add_argument("--artifact", action="append", type=parse_artifact, required=True)
    manifest.add_argument("--build-input", action="append", type=parse_key_value, default=[])
    manifest.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "manifest":
            manifest = create_manifest(
                args.jni_sha,
                args.cudf_sha,
                args.logical_version,
                args.artifact,
                args.build_input,
            )
            write_json(args.output, manifest)
        else:
            manifest = json.loads(args.manifest.read_text())
            verify_manifest(manifest, args.root)
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
