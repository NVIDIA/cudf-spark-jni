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

"""JNI image metadata shared by publishers and consumers; no registry access."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent.parent
IMAGE = "ghcr.io/nvidia/cudf-spark-jni/jni-build"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with (ROOT / path).open(encoding="utf-8") as source:
        return json.load(source)


def recipe_fingerprint(config):
    helper = os.environ.get("CI_IMAGE_METADATA_HELPER")
    require(helper, "Set CI_IMAGE_METADATA_HELPER to the common container-image/metadata.py helper")
    path = Path(helper).resolve()
    require(path.is_file(), f"Image metadata helper not found: {path}")
    spec = importlib.util.spec_from_file_location("ci_image_metadata", path)
    require(spec and spec.loader, f"Cannot load image metadata helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fingerprint(config, ROOT)


def project_version():
    version = ET.parse(ROOT / "pom.xml").getroot().findtext(
        "{http://maven.apache.org/POM/4.0.0}version"
    )
    require(version and re.fullmatch(r"[0-9]{2}\.[0-9]{2}\.[0-9]+(-[A-Za-z0-9.-]+)?", version),
            "pom.xml must declare a literal YY.MM.patch project version")
    return {"version": version, "release-line": version[:5]}


def selected_toolchains(line, allow_main):
    pins = read_json("ci/toolchains.json")
    require(isinstance(pins, dict), "Toolchain pins must be an object")
    key = line if line in pins else "main" if allow_main else line
    require(key in pins, f"Missing toolchain pins for {line}; development callers may explicitly "
            "allow main fallback, release callers require exact version pins")
    return key, pins[key]


def version_context(branch):
    metadata = project_version()
    branch = branch.removeprefix("refs/heads/")
    line = metadata["release-line"]
    require(not branch.startswith("release/") or branch == f"release/{line}",
            f"Source branch {branch} does not match project release line {line}")
    key, toolchains = selected_toolchains(line, branch == "main")
    metadata.update({"source-branch": branch, "toolchain-key": key,
                     "tag-prefix": "main" if branch == "main" else line})
    return metadata, toolchains


def image_catalog():
    catalog = read_json(".github/image-builds.json")
    require(isinstance(catalog, dict) and catalog, "Image catalog must be a non-empty object")
    require(all(isinstance(entry, dict) for entry in catalog.values()), "Invalid image catalog entry")
    return catalog


def validate_toolchains(toolchains, families):
    require(isinstance(toolchains, dict) and set(toolchains) == set(families),
            "Selected toolchains must match the image catalog's CUDA families")
    for family, cuda in toolchains.items():
        require(isinstance(family, str) and re.fullmatch(r"cuda[0-9]+", family) and
                isinstance(cuda, str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", cuda) and
                family == f"cuda{cuda.split('.')[0]}", f"Invalid CUDA toolchain {family}={cuda}")


def describe(entry, toolchains, prefix):
    config = copy.deepcopy(entry)
    family = config["toolchain"]
    require(family in toolchains, f"Missing toolchain {family}")
    cuda = toolchains[family]
    validate_toolchains({family: cuda}, [family])
    args = config["build-args"]
    require("CUDA_VERSION" not in args, "Set CUDA_VERSION only in ci/toolchains.json, not the catalog")
    os_release = args["OS_RELEASE"]
    require(isinstance(os_release, str) and re.fullmatch(r"[0-9]+", os_release), "Invalid OS_RELEASE")
    platform = config["platform"]
    require(platform in ("linux/amd64", "linux/arm64"), f"Unsupported image platform {platform}")
    arch = platform.split("/")[1]
    require(args["TARGETPLATFORM"] == platform and
            args["CMAKE_ARCH"] == ("aarch64" if arch == "arm64" else "x86_64"),
            "TARGETPLATFORM and CMAKE_ARCH must match image platform")
    require(prefix == "main" or re.fullmatch(r"[0-9]{2}\.[0-9]{2}", prefix), "Invalid image tag prefix")
    parallel = os.environ.get("PARALLEL_LEVEL", str(os.cpu_count() or 1))
    require(re.fullmatch(r"[1-9][0-9]*", parallel), "PARALLEL_LEVEL must be positive")
    args.update(CUDA_VERSION=cuda, PARALLEL_LEVEL=str(min(int(parallel), 8)))
    for key in ("cache-from", "cache-to"):
        config[key] = [value.replace("${CUDA_VERSION}", cuda) for value in config.get(key, [])]
    fingerprint = recipe_fingerprint(config)
    manifest_tag = f"{prefix}-cuda{cuda}-rockylinux{os_release}"
    config.update(version=fingerprint, tag=f"{manifest_tag}-{arch}")
    config["manifest-tag"] = manifest_tag
    config["smoke-args"] = [cuda, fingerprint]
    return config


def select(classifiers, allow_main=False):
    metadata = project_version()
    key, toolchains = selected_toolchains(metadata["release-line"], allow_main)
    catalog = image_catalog()
    validate_toolchains(toolchains, {entry["toolchain"] for entry in catalog.values()})
    if classifiers is None:
        classifiers = [entry["toolchain"] + ("-arm64" if entry["platform"] == "linux/arm64" else "")
                       for entry in catalog.values()]
    else:
        classifiers = classifiers.split(",")
    result = {}
    for classifier in dict.fromkeys(classifiers):
        require(re.fullmatch(r"cuda[0-9]+(-arm64)?", classifier), f"Invalid classifier {classifier}")
        family = classifier.split("-")[0]
        platform = "linux/arm64" if classifier.endswith("-arm64") else "linux/amd64"
        entries = [entry for entry in catalog.values() if entry["image"] == "jni-build" and
                   entry["toolchain"] == family and entry["platform"] == platform]
        require(len(entries) == 1, f"Expected one image recipe for {classifier}")
        config = describe(entries[0], toolchains, metadata["release-line"])
        result[classifier] = {
            "image": IMAGE, "platform": platform, "version": config["version"],
            "tags": [config["manifest-tag"]] + (
                ["main-" + config["manifest-tag"].split("-", 1)[1]] if allow_main else []),
            "project-version": metadata["version"], "release-line": metadata["release-line"],
            "toolchain-key": key, "cuda-version": toolchains[family],
            "config": config,
        }
    return result


def manifest_sources(tag, directory, source_sha):
    """Require both verified platform records from this publisher's source revision."""
    require(source_sha and re.fullmatch(r"[0-9a-f]{40}", source_sha), "Invalid source SHA")
    configs = json.loads(os.environ["CI_IMAGE_CONFIGS"])
    expected = [config for config in configs if config["manifest-tag"] == tag]
    platforms = ["linux/amd64", "linux/arm64"]
    require(len(expected) == 2 and sorted(config["platform"] for config in expected) == platforms,
            "A JNI manifest requires exactly one amd64 and one arm64 recipe")
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(Path(directory).glob("*.json"))]
    require(len(records) == 2, "Both native builds must publish verified image records")
    require(sorted(record["platform"] for record in records) == platforms,
            "Missing or duplicate image platform")
    sources = []
    for config in sorted(expected, key=lambda config: config["platform"]):
        record = next(record for record in records if record["platform"] == config["platform"])
        require(record["source-sha"] == source_sha and record["tag"] == config["tag"] and
                record["version"] == config["version"],
                "Image record does not match this source revision and platform recipe")
        reference = record["reference"]
        require(isinstance(reference, str) and
                re.fullmatch(re.escape(IMAGE) + r"@sha256:[0-9a-f]{64}", reference),
                "Image record must contain a verified immutable JNI image reference")
        sources.append(reference)
    require(len(set(sources)) == 2, "Platforms must have distinct image digests")
    return {"image": IMAGE, "tag": tag, "platforms": platforms, "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("version", "list", "describe", "matrix", "select", "manifest"))
    parser.add_argument("entry", nargs="?")
    parser.add_argument("--records-dir", help="Verified native publication records (manifest only)")
    parser.add_argument("--source-sha", help="Expected publisher source revision (manifest only)")
    parser.add_argument("--branch", help="Publisher source/PR base branch (version command only)")
    parser.add_argument("--classifiers", help="Comma-separated CUDA classifiers; defaults to all catalog entries")
    parser.add_argument("--allow-main-fallback", action="store_true",
                        help="Development consumers only: permit main pins and a matching main image")
    args = parser.parse_args()
    if args.entry is not None and args.operation not in ("describe", "manifest"):
        parser.error("Only describe or manifest accepts an entry")
    if args.operation == "manifest":
        if not all((args.entry, args.records_dir, args.source_sha)):
            parser.error("manifest requires a tag, --records-dir and --source-sha")
    elif args.records_dir is not None or args.source_sha is not None:
        parser.error("--records-dir and --source-sha require manifest")
    if args.branch is not None and args.operation != "version":
        parser.error("--branch is for publisher version metadata, not consumer image selection")
    if (args.classifiers is not None or args.allow_main_fallback) and args.operation != "select":
        parser.error("--classifiers and --allow-main-fallback require select")
    if args.operation == "manifest":
        result = manifest_sources(args.entry, args.records_dir, args.source_sha)
    elif args.operation == "version":
        result = project_version() if args.branch is None else version_context(args.branch)[0]
    elif args.operation == "select":
        result = select(args.classifiers, args.allow_main_fallback)
    else:
        catalog = image_catalog()
        if args.operation == "list":
            result = sorted(catalog)
        else:
            require("CI_TOOLCHAINS" in os.environ, "Pass the selected CI_TOOLCHAINS map")
            toolchains = json.loads(os.environ["CI_TOOLCHAINS"])
            prefix = os.environ.get("IMAGE_TAG_PREFIX", project_version()["release-line"])
            if args.operation == "describe":
                require(args.entry in catalog, f"Unknown catalog entry {args.entry}")
                result = describe(catalog[args.entry], toolchains, prefix)
            else:
                validate_toolchains(toolchains, {entry["toolchain"] for entry in catalog.values()})
                images = {}
                for name in sorted(catalog):
                    config = describe(catalog[name], toolchains, prefix)
                    key = config["image"] + ":" + config["tag"]
                    require(key not in images or images[key]["version"] == config["version"],
                            f"Conflicting recipes for {key}")
                    images.setdefault(key, config)
                require(0 < len(images) <= 256, "Image matrix must contain 1 to 256 entries")
                result = [images[key] for key in sorted(images)]
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, ET.ParseError) as error:
        sys.exit(str(error))
