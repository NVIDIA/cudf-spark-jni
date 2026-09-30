#!/bin/bash

#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

# Last tested: 
# Edition	Windows 10 Enterprise
# Version	21H2
# OS build	19044.1645
# Experience	Windows Feature Experience Pack 120.2212.4170.0
# NVIDIA Display Driver 473.47

# add WSL2 user to passwordless sudoers if desired
# sudo visudo /etc/sudoers.d/wsl2-sudo

# Docker 
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -a -G docker $USER
sudo service docker start
docker run hello-world

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo service docker restart


# CUDA
# Initial instructions
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network
distroArch="wsl-ubuntu/x86_64"
wget https://developer.download.nvidia.com/compute/cuda/repos/${distroArch}/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
# Instructions for fetching keys modified per
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772#install-new-cuda-keyring-package-3
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/${distroArch}/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${distroArch}/ /"
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-7