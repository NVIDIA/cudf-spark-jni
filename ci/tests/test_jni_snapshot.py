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

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


MODULE_PATH = Path(__file__).parents[1] / "jni_snapshot.py"
SPEC = importlib.util.spec_from_file_location("jni_snapshot", MODULE_PATH)
jni_snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jni_snapshot)


JNI_SHA = "a" * 40
CUDF_SHA = "b" * 40
VERSION = "26.10.0-SNAPSHOT"


def make_runtime_jar(path, jni_sha=JNI_SHA, cudf_sha=CUDF_SHA):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            jni_snapshot.JNI_PROPERTIES,
            f"version={VERSION}\nrevision={jni_sha}\n",
        )
        archive.writestr(
            jni_snapshot.CUDF_PROPERTIES,
            f"version={VERSION}\nrevision={cudf_sha}\n",
        )


class JniSnapshotTest(unittest.TestCase):
    def test_manifest_uses_moving_version_and_embedded_source(self):
        with tempfile.TemporaryDirectory() as directory:
            jar = Path(directory) / f"cudf-spark-jni-{VERSION}-cuda12.jar"
            make_runtime_jar(jar)
            manifest = jni_snapshot.create_manifest(
                JNI_SHA,
                CUDF_SHA,
                VERSION,
                [("cuda12", "jar", jar)],
                [("image.cuda12", "sha256:image"), ("cmake", "4.2.3")],
            )
            self.assertEqual(VERSION, manifest["logicalVersion"])
            self.assertEqual(JNI_SHA, manifest["source"]["jniSha"])
            self.assertEqual(64, len(manifest["build"]["fingerprint"]))
            self.assertEqual(jni_snapshot.digest(jar, "sha256"),
                             manifest["artifacts"][0]["sha256"])

    def test_rejects_commit_qualified_version(self):
        with self.assertRaisesRegex(ValueError, "moving SNAPSHOT"):
            jni_snapshot.validate_version("26.10.0-jni.a33da84b12ef.r1-SNAPSHOT")

    def test_rejects_runtime_jar_from_another_jni_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            jar = Path(directory) / "jni.jar"
            make_runtime_jar(jar, jni_sha="d" * 40)
            with self.assertRaisesRegex(ValueError, "embeds JNI revision"):
                jni_snapshot.create_manifest(
                    JNI_SHA,
                    CUDF_SHA,
                    VERSION,
                    [("cuda12", "jar", jar)],
                    [],
                )

    def test_verifies_retained_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jar = root / "jni.jar"
            make_runtime_jar(jar)
            manifest = jni_snapshot.create_manifest(
                JNI_SHA,
                CUDF_SHA,
                VERSION,
                [("cuda12", "jar", jar)],
                [("cuda", "12.9.1")],
            )
            jni_snapshot.verify_manifest(manifest, root)

    def test_detects_changed_retained_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jar = root / "jni.jar"
            make_runtime_jar(jar)
            manifest = jni_snapshot.create_manifest(
                JNI_SHA,
                CUDF_SHA,
                VERSION,
                [("cuda12", "jar", jar)],
                [],
            )
            jar.write_bytes(b"changed")
            with self.assertRaises((ValueError, zipfile.BadZipFile)):
                jni_snapshot.verify_manifest(manifest, root)


if __name__ == "__main__":
    unittest.main()
