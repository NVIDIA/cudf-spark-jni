# Contributing to RAPIDS Accelerator JNI for Apache Spark

Contributions to RAPIDS Accelerator JNI for Apache Spark fall into the following three categories.

1. To report a bug, request a new feature, or report a problem with
    documentation, please file an [issue](https://github.com/NVIDIA/spark-rapids-jni/issues/new/choose)
    describing in detail the problem or new feature. The project team evaluates
    and triages issues, and schedules them for a release. If you believe the
    issue needs priority attention, please comment on the issue to notify the
    team.
2. To propose and implement a new Feature, please file a new feature request
    [issue](https://github.com/NVIDIA/spark-rapids-jni/issues/new/choose). Describe the
    intended feature and discuss the design and implementation with the team and
    community. Once the team agrees that the plan looks good, go ahead and
    implement it using the [code contributions](#code-contributions) guide below.
3. To implement a feature or bug-fix for an existing outstanding issue, please
    follow the [code contributions](#code-contributions) guide below. If you
    need more context on a particular issue, please ask in a comment.

## Branching Convention

There are two types of branches in this repository:

* `branch-[version]`: are development branches which can change often. Note that we merge into
  the branch with the greatest version number, as that is our default branch.

* `main`: is the branch with the latest released code, and the version tag (i.e. `v22.02.0`)
  is held here. `main` will change with new releases, but otherwise it should not change with
  every pull request merged, making it a more stable branch.

## Git Submodules

This repository uses Git submodules. After cloning this repository or moving to a new commit
in this repository you will need to ensure the submodules are initialized and updated to the
expected submodule commits. This can be done by executing the following command at the top of
the repository:

```commandline
git submodule update --init --recursive
```

## Building From Source

[Maven](https://maven.apache.org) is used for most aspects of the build. For example, the
Maven `package` goal can be used to build the RAPIDS Accelerator JNI jar. After a successful
build the RAPIDS Accelerator JNI jar will be in the `spark-rapids-jni/target/` directory.
Be sure to select the jar with the CUDA classifier.

When building spark-rapids-jni, the pom.xml in the submodule thirdparty/cudf is completely
bypassed. For a detailed explanation please read
[this](https://github.com/NVIDIA/spark-rapids-jni/issues/1084#issuecomment-1513471739).

### Building in the Docker Container

The `build/build-in-docker` script will build the spark-rapids-jni artifact within a Docker
container using devtoolset to produce native code that can run on all supported Linux
distributions. The repo directory is bind-mounted into the container and the container runs
as the current user, so the artifacts are produced as if they were built or installed outside
the Docker container.

The script passes all of its arguments onto the Maven command run inside the Docker container,
so it should be invoked as one would invoke Maven, e.g.: `build/build-in-docker clean package`

#### Using spark-rapids-jni Docker Container with other Repos

Spark RAPIDS project spans multiple repos. Some issues are discovered in
spark-rapids-jni but they need to be made easily reproducible in the cudf repo

To this end export WORKDIR with the path pointing to a different repo

```
export WORKDIR=~/gits/rapidsai/cudf
~/gits/NVIDIA/spark-rapids-jni/build/run-in-docker head README.md
```

### cudf Submodule and Build

[RAPIDS cuDF](https://github.com/rapidsai/cudf) is being used as a submodule in this project.

Currently libcudf is only configured once and the build relies on cmake to re-configure as needed.
This is because libcudf currently is rebuilding almost entirely when it is configured with the same
settings. If an explicit reconfigure of libcudf is needed (e.g.: when changing compile settings via
`CMAKE_CUDA_ARCHITECTURES`, `CUDF_USE_PER_THREAD_DEFAULT_STREAM`, etc.) then a configure can be forced via
`-Dlibcudf.build.configure=true`.

### Build Properties

The following build properties can be set on the Maven command-line (e.g.: `-DCPP_PARALLEL_LEVEL=4`)
to control aspects of the build:

| Property Name                        | Description                             | Default |
|--------------------------------------|-----------------------------------------|---------|
| `CPP_PARALLEL_LEVEL`                 | Parallelism of the C++ builds           | 10      |
| `CMAKE_CUDA_ARCHITECTURES`           | CUDA architectures to target            | RAPIDS  |
| `CUDF_USE_PER_THREAD_DEFAULT_STREAM` | CUDA per-thread default stream          | ON      |
| `RMM_LOGGING_LEVEL`                  | RMM logging control                     | OFF     |
| `USE_GDS`                            | Compile with GPU Direct Storage support | OFF     |
| `BUILD_TESTS`                        | Compile tests                           | OFF     |
| `BUILD_BENCHMARKS`                   | Compile benchmarks                      | OFF     |
| `BUILD_FAULTINJ`                     | Compile fault injection                 | ON      |
| `libcudf.build.configure`            | Force libcudf build to configure        | false   |
| `submodule.check.skip`               | Whether to skip checking git submodules | false   |


### Local testing of cross-repo contributions cudf, spark-rapids-jni, and spark-rapids

When we work on a feature or a bug fix across repositories, it is beneficial to be able to
run manual and integration tests end to end on the full stack from Apache Spark
with spark-rapids plugin upfront before merging the PRs.

So we are dealing with a subset of the following:

Local PR branches for
- rapidsai/cuDF, branch pr1
- NVIDIA/spark-rapids-jni, branch pr2
- NVIDIA/spark-rapids, branch pr3

Our end goal is to build the rapids-4-spark dist jar in the pr3 branch under local repo path
~/repos/NVIDIA/spark-rapids that includes changes from the pr2 branch in
~/repos/NVIDIA/spark-rapids-jni and the pr1 branch in rapidsai/cuDF that we will test
with Spark. There are two options for working on pr1.

#### Option 1: Working on cuDF PR inside the the submodule in spark-rapids-jni
To avoid retargeting the submodule to the local cuDF repo as below, we might find it easier
to make changes locally under ~/repos/NVIDIA/spark-rapids-jni/thirdparty/cudf directly.

In order to push pr1 to create a pull request, we need to add a remote to the submodule for the cuDF
fork in our account

```bash
$ cd ~/repos/NVIDIA/spark-rapids-jni/thirdparty/cudf
$ git remote add <user> git@github.com:<user>/cudf.git
# make and commit changes
$ git push <user>
```

#### Option 2: Working on cuDF PR in a conventional local cuDF fork
Once we are done with our changes to the pr1 branch in
~/repos/rapidsai/cuDF, we git commit changes locally.

Then we cd to ~/repos/NVIDIA/spark-rapids-jni and point the cudf submodule temporarily to the pr1
branch

```bash
$ git submodule set-url thirdparty/cudf ~/repos/rapidsai/cudf
$ git submodule set-branch --branch pr1 thirdparty/cudf
```

Sync pr1 into our pr2 branch in ~/repos/NVIDIA/spark-rapids-jni
```bash
$ git submodule sync --recursive
$ git submodule update --init --recursive --remote
```

#### Building final spark-rapids artifact with pr1, pr2, and pr3 changes
Regardless what option we have used to make cuDF changes, we proceed with building
spark-rapids-jni. The spark-rapids repo will consume spark-rapids-jni with pr1 and pr2 changes
from the local Maven cache after we run `mvn install` via `build/build-in-docker`
in ~/repos/NVIDIA/spark-rapids-jni.

Make sure to stage thirdparty/cudf with `git add` to satifsfy build's submodule check.
```bash
$ git add thirdparty/cudf
$ ./build/build-in-docker install ...
```

Now cd to ~/repos/NVIDIA/spark-rapids and build with one of the options from
[spark-rapids instructions](https://github.com/NVIDIA/spark-rapids/blob/main/CONTRIBUTING.md#building-from-source).

```bash
$ ./build/buildall
```

Since we rely on local Maven cache we need to pay extra attention to make sure that
the final rapids-4-spark artifact includes the locally built dependencies as opposed to
CI-built snapshot dependencies from the remote Maven repo. This may happen even if Maven
is invoked with `--offline` or `--no-snapshot-updates` option due to IDE-Maven
interactions in the background. To confirm that the artifact is correct we can either enable
[INFO logging in Spark](https://github.com/NVIDIA/spark-rapids/blob/4c77f0db58d229b2e6cb75c196934fcc0ae3a485/sql-plugin/src/main/scala/com/nvidia/spark/rapids/Plugin.scala#L73-L83)
or directly inspect the resulting jar for build info:
```bash
$ unzip -c dist/target/rapids-4-spark_2.12-22.08.0-SNAPSHOT-cuda12.jar *version-info.properties
Archive:  dist/target/rapids-4-spark_2.12-22.08.0-SNAPSHOT-cuda12.jar
  inflating: cudf-java-version-info.properties
version=22.08.0-SNAPSHOT
user=
revision=62657ad6a296ea3547417504652e3b8836b020fb
branch=testCUDF_pr1
date=2022-07-19T21:48:15Z
url=https://github.com/rapidsai/cudf.git

  inflating: spark-rapids-jni-version-info.properties
version=22.08.0-SNAPSHOT
user=
revision=70adcc86a513ad6665968021c669fbca7515a188
branch=pr/user1/381
date=2022-07-19T21:48:15Z
url=git@github.com:NVIDIA/spark-rapids-jni.git

  inflating: rapids4spark-version-info.properties
version=22.08.0-SNAPSHOT
cudf_version=22.08.0-SNAPSHOT
user=user1
revision=6453047ef479b5ec79384c5150c50af2f50f563e
branch=aqeFinalPlanOnGPUDoc
date=2022-07-19T21:51:52Z
url=https://github.com/NVIDIA/spark-rapids
```
and verify that the branch names and the revisions in the console output
correspond the local repos.

When we are ready to move on, prior to switching to another spark-rapids-jni branch
or submiting a PR to NVIDIA/spark-rapids-jni, we should undo the cudf submodule modifications.
```
$ cd ~/repos/NVIDIA/spark-rapids-jni
$ git restore .gitmodules
$ git restore --staged thirdparty/cudf
```

### Building on Windows in WSL2
Building on Windows can be done if your Windows build version supports
[WSL2](https://docs.microsoft.com/en-us/windows/wsl/install). You can create a minimum
Ubuntu distro WSL2 instance to be able to run `build/build-in-docker` above.
```PowerShell
> wsl --install -d Ubuntu
> .\build\win\create-wsl2.ps1
```

Clone spark-rapids-jni inside or outside (convenient but slower filesystem) the distro,
and build inside WSL2, e.g.
```PowerShell
> wsl -d Ubuntu ./build/build-in-docker clean install -DGPU_ACRCHS=NATIVE -Dtest="*,!CuFileTest"
```

### Testing
Java tests are in the `src/test` directory and c++ tests are in the `src/main/cpp/tests` directory.
The c++ tests are built with the `-DBUILD_TESTS` command line option and will build into the
`target/jni/cmake-build/gtests/` directory. Due to building inside the docker container, it is possible
that the host environment does not match the container well enough to run these executables, resulting
in errors finding libraries. The script `build/run-in-docker` was created to help with this
situation. A test can be run directly using this script or the script can be run without any
arguments to get into an interactive shell inside the container.
```
build/run-in-docker target/jni/cmake-build/gtests/ROW_CONVERSION
```

The full C++ test suite should be run on a GPU with enough free device memory for the largest
test inputs and conversion temporaries. For example, `RowToColumnTests.Biggest` builds a
128-column table with `2 * 1024 * 1024` `INT32` rows, which is about 1 GiB of input data before
the row-conversion output and intermediate allocations. This test is known to run out of memory
on 6 GiB GPUs. When iterating on smaller development GPUs, run a narrower gtest filter, such as:
```
build/run-in-docker target/jni/cmake-build/gtests/ROW_CONVERSION \
  --gtest_filter=-RowToColumnTests.Biggest
```
Use a larger-memory GPU when validating the complete C++ test suite.

#### Testing with Compute Sanitizer
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) is a
functional correctness checking suite included in the CUDA toolkit. The RAPIDS Accelerator JNI
supports leveraging the Compute Sanitizer in memcheck mode in the unit tests to help catch any kernels
that may be doing something incorrectly. To run the unit tests with the Compute Sanitizer, append the
`-DUSE_SANITIZER=ON` to the build command. e.g.
```
>  ./build/build-in-docker clean package -DUSE_SANITIZER=ON
```

The Compute Sanitizer will output its report into one or multiple log files named as
`sanitizer_for_pid_<pid number>.log` under the current workspace root path.

Please note not all the unit tests can run with Compute Sanitizer. For example, `RmmTest#testEventHandler`,
a problematic test, intentionally tries an illegal allocation because of a too big size as part of the
test, but Compute Sanitizer will still report the errors and fail the whole build process.
`UnsafeMemoryAccessorTest` is for host memory only, so there is no need to run it with
Compute Sanitizer either.

If you think your tests are not suitable for Compute Sanitizer, please add the JUnit5 tag (`@Tag("noSanitizer")`)
to the tests or the test class.
```
@Tag("noSanitizer")
class ExceptionCaseTest { ... }

# or for a single test
class NormalCaseTest {

  @Tag("noSanitizer")
  public void testOneErrorCase(){ ... }
}
```

### Debugging
You can add debug symbols selectively to C++ files in spark-rapids-jni by modifying the appropriate
`CMakeLists.txt` files. You will need to add a specific flag depending on what kind of code you are
debugging. For CUDA code, you need to add the `-G` flag to add device debug symbols:

```cmake
set_source_files_properties(src/row_conversion.cu PROPERTIES COMPILE_OPTIONS "-G")
```

For C++ code, you will need to add the `-g` flag to add host debug symbols.

```cmake
set_source_files_properties(row_conversion.cpp PROPERTIES COMPILE_OPTIONS "-G")
```

For debugging C++ tests, you need to add both device debug symbols to the CUDA kernel files involved
in testing (in `src/main/cpp/CMakeLists.txt`) **and** host debug symbols to the CPP files used for
testing (in `src/main/cpp/tests/CMakeLists.txt`).

You can then use `cuda-gdb` to debug the gtest (NOTE: For Docker, run an interactive shell first and
then run `cuda-gdb`. You do not necessarily need to run `cuda-gdb` in Docker):

```bash
./build/run-in-docker
bash-4.2$ cuda-gdb target/jni/cmake-build/gtests/ROW_CONVERSION
```

You can also use the [NVIDIA Nsight VSCode Code Integration](https://docs.nvidia.com/nsight-visual-studio-code-edition/latest/cuda-debugger/index.html)
as well to debug within Visual Studio Code.

To debug libcudf code, please see [Debugging cuDF](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md#debugging-cudf)
in the cuDF [CONTRIBUTING](https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md) guide.

### Benchmarks
Benchmarks exist for c++ benchmarks using NVBench and are in the `src/main/cpp/benchmarks` directory.
To build these benchmarks requires the `-DBUILD_BENCHMARKS` build option. Once built, the benchmarks
can be found in the `target/jni/cmake-build/benchmarks/` directory. Due to building inside the docker
container, it is possible that the host environment does not match the container well enough to
run these executables, resulting in errors finding libraries. The script `build/run-in-docker`
was created to help with this situation. A benchmark can be run directly using this script or the
script can be run without any arguments to get into an interactive shell inside the container.
```build/run-in-docker target/jni/cmake-build/benchmarks/ROW_CONVERSION_BENCH```
## Code contributions

### Your first issue

1. Read the [Developer Overview](https://github.com/NVIDIA/spark-rapids/blob/main/docs/dev/README.md)
    to understand how the RAPIDS Accelerator plugin works.
2. Find an issue to work on. The best way is to look for the
    [good first issue](https://github.com/NVIDIA/spark-rapids-jni/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/NVIDIA/spark-rapids-jni/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
    labels.
3. Comment on the issue stating that you are going to work on it.
4. Code! Make sure to add or update unit tests if needed!
5. When done, [create your pull request](https://github.com/NVIDIA/spark-rapids-jni/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/).
    Fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a project committer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Code Formatting

#### Java

This Java code in this project (`src/main/java`) follows the
[Oracle Java code conventions](http://www.oracle.com/technetwork/java/codeconvtoc-136057.html).


#### C++

The C++ code in this project (`src/main/cpp`) follows the
[coding style from `rapidsai/cudf` repository](https://github.com/rapidsai/cudf/blob/main/cpp/doxygen/developer_guide/DEVELOPER_GUIDE.md#code-and-documentation-style-and-formatting).

We also provide a precommit-hook to format code using cudf's C++ `clang-format` style.
To use precommit-hook, install it on your system such as using `conda` or `pip`:
```
conda install -c conda-forge pre-commit
```
```
pip install pre-commit
```

Then, run pre-commit hooks before committing your code. This will reformat the staged files:
```
pre-commit run
```

And for reformatting all files:
```
pre-commit run --all-files
```

### Sign your work

We require that all contributors sign-off on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not signed off will not be accepted.

To sign off on a commit use the `--signoff` (or `-s`) option when committing your changes:

```shell
git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The sign-off is a simple line at the end of the explanation for the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. Use your real name, no pseudonyms or anonymous contributions.  If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with `git commit -s`.


The signoff means you certify the below (from [developercertificate.org](https://developercertificate.org)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Attribution
Portions adopted from https://github.com/rapidsai/cudf/blob/main/CONTRIBUTING.md, https://github.com/NVIDIA/nvidia-docker/blob/main/CONTRIBUTING.md, and https://github.com/NVIDIA/DALI/blob/main/CONTRIBUTING.md
