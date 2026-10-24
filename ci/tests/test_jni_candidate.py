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


MODULE_PATH = Path(__file__).parents[1] / "jni_candidate.py"
SPEC = importlib.util.spec_from_file_location("jni_candidate", MODULE_PATH)
jni_candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jni_candidate)


JNI_SHA = "a33da84b12ef0000000000000000000000000000"
CUDF_SHA = "c101000000000000000000000000000000000000"


class CandidateVersionTest(unittest.TestCase):
    def test_creates_unique_snapshot_version(self):
        self.assertEqual(
            "26.12.0-jni.a33da84b12ef.r2-SNAPSHOT",
            jni_candidate.candidate_version("26.12.0-SNAPSHOT", JNI_SHA, 2),
        )

    def test_rejects_short_sha(self):
        with self.assertRaisesRegex(ValueError, "full 40-character"):
            jni_candidate.candidate_version("26.12.0-SNAPSHOT", "a33da84b12ef", 1)

    def test_rejects_zero_revision(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            jni_candidate.candidate_version("26.12.0-SNAPSHOT", JNI_SHA, 0)

    def test_prepare_updates_only_project_version(self):
        with tempfile.TemporaryDirectory() as directory:
            pom = Path(directory) / "pom.xml"
            pom.write_text(
                "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n"
                "  <version>26.12.0-SNAPSHOT</version>\n"
                "  <properties><dependency.version>1.0</dependency.version></properties>\n"
                "</project>\n"
            )
            candidate = jni_candidate.prepare_pom(pom, JNI_SHA, 3)
            self.assertEqual("26.12.0-jni.a33da84b12ef.r3-SNAPSHOT", candidate)
            self.assertIn(f"<version>{candidate}</version>", pom.read_text())
            self.assertIn("<dependency.version>1.0</dependency.version>", pom.read_text())


class CandidateManifestTest(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_dir = root / "target"
            artifact_dir.mkdir()
            version = "26.12.0-jni.a33da84b12ef.r1-SNAPSHOT"
            pom = root / "pom.xml"
            pom.write_text(
                "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">"
                f"<version>{version}</version></project>"
            )
            for suffix, content in {
                "cuda12.jar": b"cuda12",
                "cuda13.jar": b"cuda13",
                "sources.jar": b"sources",
                "javadoc.jar": b"javadocs",
            }.items():
                (artifact_dir / f"cudf-spark-jni-{version}-{suffix}").write_bytes(content)

            args = type(
                "Args",
                (),
                {
                    "pom": pom,
                    "repo": root,
                    "artifact_dir": artifact_dir,
                    "artifact_id": "cudf-spark-jni",
                    "classifier": ["cuda13", "cuda12"],
                    "default_classifier": "cuda12",
                    "jni_sha": JNI_SHA,
                    "cudf_sha": CUDF_SHA,
                    "revision": 1,
                },
            )()
            manifest = jni_candidate.create_manifest(args)

            self.assertEqual(version, manifest["candidateId"])
            self.assertEqual(["cuda12", "cuda13"], manifest["requiredClassifiers"])
            self.assertEqual("cuda12", manifest["defaultClassifier"])
            self.assertEqual(6, len(manifest["artifacts"]))
            self.assertEqual(CUDF_SHA, manifest["source"]["cudfCommit"])
            self.assertEqual(
                sorted(entry["publishedName"] for entry in manifest["artifacts"]),
                [entry["publishedName"] for entry in manifest["artifacts"]],
            )
            json.dumps(manifest)

    def test_manifest_rejects_missing_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version = "26.12.0-jni.a33da84b12ef.r1-SNAPSHOT"
            pom = root / "pom.xml"
            pom.write_text(
                "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">"
                f"<version>{version}</version></project>"
            )
            args = type(
                "Args",
                (),
                {
                    "pom": pom,
                    "repo": root,
                    "artifact_dir": root,
                    "artifact_id": "cudf-spark-jni",
                    "classifier": ["cuda12"],
                    "default_classifier": "cuda12",
                    "jni_sha": JNI_SHA,
                    "cudf_sha": CUDF_SHA,
                    "revision": 1,
                },
            )()
            with self.assertRaisesRegex(ValueError, "required candidate artifacts"):
                jni_candidate.create_manifest(args)


if __name__ == "__main__":
    unittest.main()
