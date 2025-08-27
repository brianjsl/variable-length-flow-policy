#!/bin/bash

set -euo pipefail

TMP_DOCKER_BUILD_DIR="/tmp/lbm_docker_build"

# Check if an argument is provided
if [ $# -gt 2 ]; then
    echo "Usage: $0 {build|run} {20.04|22.04|24.04}"
    exit 1
fi

# Get the input argument
ACTION=$1
UBUNTU_VERSION=${2:-"20.04"}

# Validate UBUNTU_VERSION and determine the base image & requirements file.
REQUIREMENTS_FILE=requirements.in
if [ $UBUNTU_VERSION == "20.04" ]; then
  BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker
elif [ $UBUNTU_VERSION == "22.04" ]; then
  BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker
elif [ $UBUNTU_VERSION == "24.04" ]; then
  BASE_IMAGE=124224456861.dkr.ecr.us-west-2.amazonaws.com/lbm-pytorch-training:2.6.0-gpu-py312-cu126-ubuntu24.04-sagemaker
else
  echo "Only Ubuntu 20.04, 22.04 and 24.04 are supported."
  exit
fi

# Check the value of the argument
if [ "$ACTION" == "build" ]; then
    echo "Building the container..."

    # Clean up a temp folder if exits from the previously failed build.
    if [ -d ${TMP_DOCKER_BUILD_DIR} ]; then
      rm -rf ${TMP_DOCKER_BUILD_DIR}
    fi

    # Create a temp folder and copy the local lbm into it.
    mkdir ${TMP_DOCKER_BUILD_DIR}
    cd $(dirname $0)/../../../ && cp lbm ${TMP_DOCKER_BUILD_DIR} -r
    cd ${TMP_DOCKER_BUILD_DIR}
    cp lbm/setup/docker/Dockerfile .
    cp lbm/setup/docker/.dockerignore ./.dockerignore
    sed -i -e "s/REGION_PLACEHOLDER/us-east-1/g" Dockerfile
    cp lbm/${REQUIREMENTS_FILE} .
    sed -i "/^torch==/d" ${REQUIREMENTS_FILE}
    # Copy all the pip wheels
    mkdir -p setup/venv/
    cp -r lbm/setup/venv/wheels setup/venv/
    docker build -t lbm-local:latest . --build-arg local_build=TRUE --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg REQUIREMENTS_FILE=${REQUIREMENTS_FILE}
    rm -rf ${TMP_DOCKER_BUILD_DIR}

elif [ "$ACTION" == "run" ]; then
    echo "Running the container..."
    if [ -f /etc/profile.d/aws.sh ]; then
      # INFO(shun): DGX doesn't use SSO.
      bash /etc/profile.d/aws.sh
      docker container run --privileged --gpus all --shm-size=1g \
            --ipc=host --network=host \
            -e NCCL_DEBUG=INFO \
            -e NCCL_SOCKET_IFNAME=^docker0,lo -it \
            -e AWS_ACCESS_KEY_ID \
            -e AWS_SECRET_ACCESS_KEY \
            -e AWS_DEFAULT_REGION \
            -e WANDB_API_KEY \
            -e WANDB_MODE \
            -e WANDB_ENTITY \
            -v $HOME/.aws:/root/.aws:ro \
            --rm lbm-local:latest /bin/bash
    else
      docker container run --privileged --gpus all --shm-size=1g \
            --ipc=host --network=host \
            -e NCCL_DEBUG=INFO \
            -e NCCL_SOCKET_IFNAME=^docker0,lo -it \
            -e AWS_CONFIG_FILE=/root/.aws/config \
            -e AWS_SSO_SESSION=manip-cluster \
            -e WANDB_API_KEY \
            -e WANDB_MODE \
            -e WANDB_ENTITY \
            -v $HOME/.aws:/root/.aws:ro \
            --rm lbm-local:latest /bin/bash
    fi
else
    echo "Invalid argument: $ACTION"
    echo "Usage: $0 {build|run}"
    exit 1
fi
