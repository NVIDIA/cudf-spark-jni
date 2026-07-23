#
# Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
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
#

###
# Build the image for spark-rapids-jni CICD environment.
#
# Arguments: CUDA_VERSION=[12.X.Y], OS_RELEASE=[8, 9], TARGETPLATFORM=[linux/amd64, linux/arm64]
#
###
ARG CUDA_VERSION=12.9.1
ARG OS_RELEASE=8
ARG TARGETPLATFORM=linux/amd64

# multi-platform build with: docker buildx build --platform linux/arm64,linux/amd64 <ARGS> on either amd64 or arm64 host
# check available official arm-based docker images at https://hub.docker.com/r/nvidia/cuda/tags (OS/ARCH)
FROM --platform=$TARGETPLATFORM nvidia/cuda:$CUDA_VERSION-devel-rockylinux$OS_RELEASE

# Dependency versions
# Act as default GCC toolset in the image
ARG TOOLSET_VERSION=14
ARG CMAKE_VERSION=4.0.3
ARG CCACHE_VERSION=4.11.2

# Make it available at runtime and to the inherited images (e.g. Dockerfile.devel)
ENV TOOLSET_VERSION=${TOOLSET_VERSION}

# Default x86_64 from x86 build, aarch64 cmake for arm build
ARG CMAKE_ARCH=x86_64

### Install basic requirements
# pin urllib3<2.0 for https://github.com/psf/requests/issues/6432
RUN dnf --enablerepo=powertools install -y scl-utils gcc-toolset-13 gcc-toolset-${TOOLSET_VERSION} python39 \
    zlib-devel maven tar wget patch ninja-build git zip && \
  alternatives --set python /usr/bin/python3 && \
  python -m pip install requests 'urllib3<2.0'

# Execute every time a new non-interactive bash shell is started
ENV BASH_ENV=/etc/bashrc

## pre-create the CMAKE_INSTALL_PREFIX folder, set writable by any user for Jenkins
RUN mkdir -m 777 /usr/local/rapids /rapids

RUN cd /usr/local && wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz && \
   tar zxf cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz && \
   rm cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz

# Make version-less alias for external reference such as when cmake is called by an IDE outside of the container
RUN ln -s /usr/local/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}/bin/cmake /usr/local/bin/cmake

# ccache for interactive builds
ARG PARALLEL_LEVEL
RUN cd /tmp && wget --quiet https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}.tar.gz && \
   tar zxf ccache-${CCACHE_VERSION}.tar.gz && \
   rm ccache-${CCACHE_VERSION}.tar.gz && \
   cd ccache-${CCACHE_VERSION} && \
   mkdir build && \
   cd build && \
   scl enable gcc-toolset-${TOOLSET_VERSION} \
      "cmake .. \
         -DCMAKE_BUILD_TYPE=Release \
         -DZSTD_FROM_INTERNET=ON \
         -DREDIS_STORAGE_BACKEND=OFF && \
      cmake --build . --parallel ${PARALLEL_LEVEL} --target install" && \
   cd ../.. && \
   rm -rf ccache-${CCACHE_VERSION}

## install a version of boost that is needed for arrow/parquet to work
RUN cd /usr/local && wget --quiet https://archives.boost.io/release/1.79.0/source/boost_1_79_0.tar.gz && \
  echo "273f1be93238a068aba4f9735a4a2b003019af067b9c183ed227780b8f36062c  boost_1_79_0.tar.gz" | sha256sum -c - && \
  tar -xzf boost_1_79_0.tar.gz && \
  rm boost_1_79_0.tar.gz && \
  cd boost_1_79_0 && \
  ./bootstrap.sh --prefix=/usr/local && \
  ./b2 install --prefix=/usr/local --with-filesystem --with-system && \
   cd /usr/local && \
   rm -rf boost_1_79_0

# disable cuda container constraints to allow running w/ elder drivers on data-center GPUs
ENV NVIDIA_DISABLE_REQUIRE="true"
