#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print bounded JNI diagnostics to the worker console and GitHub job summary."""

import html
import os
from pathlib import Path
import xml.etree.ElementTree as ET


def tail(path, lines=80):
    try:
        with Path(path).open("rb") as source:
            source.seek(0, 2)
            source.seek(max(0, source.tell() - 16384))
            return "\n".join(source.read().decode("utf-8", errors="replace").splitlines()[-lines:])
    except OSError as error:
        return f"Unavailable: {path} ({error.strerror})"


def code(value):
    return f"<code>{html.escape(value or 'unavailable').replace('|', '&#124;')}</code>"


def main():
    outcome = os.environ.get("TEST_OUTCOME", "unknown")
    phase = tail("target/gha-phase.txt", 1)
    exit_code = tail("target/gha-exit-code.txt", 1)
    rows = [
        ("Workload outcome", outcome),
        ("Last phase", phase),
        ("Exit code (124 = workload deadline)", exit_code),
        ("Source", os.environ.get("GITHUB_SHA")),
        ("CUDA family", os.environ.get("CLASSIFIER")),
        ("Published image", os.environ.get("IMAGE_REFERENCE") or
         ("local recipe build" if os.environ.get("IMAGE_ID") else "unavailable")),
        ("Local image ID", os.environ.get("IMAGE_ID")),
        ("Recipe", os.environ.get("IMAGE_RECIPE")),
        ("Maven cache restored", os.environ.get("MAVEN_CACHE_KEY") or "none; see restore step"),
        ("Native cache restored", os.environ.get("NATIVE_CACHE_KEY") or "none; see restore step"),
    ]
    report = ["## JNI GPU CI diagnostics", "", "| Item | Result |", "| --- | --- |"]
    report.extend(f"| {name} | {code(value)} |" for name, value in rows)
    evidence = os.environ.get("EVIDENCE_URL")
    report += ["", f"[Download full JNI reports and logs]({evidence})" if evidence else
               "Evidence upload unavailable; check the upload step and worker console."]
    if outcome != "success":
        report += ["", "Check the failed setup step or last workload phase below. "
                   "A workload failure is not retried on another GPU."]

    failures = []
    warnings = []
    for path in sorted(Path("target/surefire-reports").glob("TEST-*.xml")):
        try:
            for _, element in ET.iterparse(path, events=("end",)):
                if element.tag != "testcase":
                    continue
                errors = element.findall("failure") + element.findall("error")
                if errors and len(failures) < 10:
                    name = f"{element.get('classname', '')}.{element.get('name', '')}"
                    message = errors[0].get("message") or errors[0].text or "See full report"
                    failures.append(f"{name[:300]}: {message[:2000]}")
                element.clear()
        except (ET.ParseError, OSError) as error:
            warnings.append(f"Cannot read {path.name}: {str(error)[:300]}")

    sections = [("GPU / driver", tail("target/gha-gpu.txt", 20))]
    if failures:
        sections.append(("Failing tests (first 10)", "\n\n".join(failures)))
    if warnings:
        sections.append(("Incomplete test reports", "\n".join(warnings[:10])))
    if outcome != "success":
        sections.append(("Workload log tail (last 80 lines)", tail("target/gha-premerge.log")))
        sections.append(("Container cleanup (best-effort; an auto-removed container may be absent)",
                         tail("target/gha-cleanup.log", 10)))
    sections.append(("ccache statistics", tail("target/gha-ccache.log", 40)))
    report += ["", "Excerpts are bounded; full test, fuzz, CMake and JVM logs are in the artifact."]
    for name, text in sections:
        report += ["", f"### {name}", "", f"<pre>{html.escape(text)}</pre>"]
    summary = "\n".join(report) + "\n"
    # Keep useful evidence in the console even if summary publication fails.
    print("=== JNI GPU CI diagnostics ===")
    for name, value in rows:
        print(f"{name}: {value or 'unavailable'}")
    print(f"Full reports: {evidence or 'unavailable; see upload step'}")
    for name, text in sections:
        print(f"\n=== {name} ===\n{text}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
            output.write(summary)


if __name__ == "__main__":
    main()
