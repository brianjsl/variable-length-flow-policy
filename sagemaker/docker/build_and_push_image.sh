#!/bin/bash
# TODO(zachfang): Consider consolidating this folder to `//lbm_sagemaker`.

set -euo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE}) && pwd)
lbm_dir=$(cd ${script_dir}/../.. && pwd)

# Default build parameters when a user doesn't specify anything.
FORK_NAME=brianjsl
BRANCH_NAME=main
REGION=us-west-2
UBUNTU_VERSION=20.04
IMAGE_TAG=latest
IMAGE_NAME=""

# Parse arguments and overwrite the default values if provided.
for named_arg in "$@"
do
  KEY=$(echo ${named_arg} | cut -f1 -d=)
  if [ $KEY != "BRANCH_NAME" ] && [ $KEY != "FORK_NAME" ] && [ $KEY != "REGION" ] && [ $KEY != "UBUNTU_VERSION" ] && [ $KEY != "IMAGE_NAME" ] && [ $KEY != "IMAGE_TAG" ]; then
    echo "Only BRANCH_NAME/FORK_NAME/REGION/UBUNTU_VERSION/IMAGE_NAME/IMAGE_TAG (case sensitive) is allowed as an argument."
    exit
  fi

  KEY_LENGTH=${#KEY}
  VALUE="${named_arg:$KEY_LENGTH+1}"
  declare "$KEY"="$VALUE"
done

# Validate `REGION`. Git will check `FORK_NAME` and `BRANCH_NAME` for us.
if [ $REGION != "us-west-2" ]; then
  echo "Only us-west-2 are supported."
  exit
fi

# Validate UBUNTU_VERSION and determine the base image & requirements file.
REQUIREMENTS_FILE=requirements.txt
if [ $UBUNTU_VERSION == "20.04" ]; then
  DOCKER_REGISTRY=763104351884.dkr.ecr.${REGION}.amazonaws.com
  BASE_IMAGE=${DOCKER_REGISTRY}/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker
elif [ $UBUNTU_VERSION == "22.04" ]; then
  DOCKER_REGISTRY=763104351884.dkr.ecr.${REGION}.amazonaws.com
  BASE_IMAGE=${DOCKER_REGISTRY}/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker
elif [ $UBUNTU_VERSION == "24.04" ]; then
  DOCKER_REGISTRY=124224456861.dkr.ecr.${REGION}.amazonaws.com
  BASE_IMAGE=${DOCKER_REGISTRY}/lbm-pytorch-training:2.6.0-gpu-py312-cu126-ubuntu24.04-sagemaker
else
  echo "Only Ubuntu 20.04, 22.04 and 24.04 are supported."
  exit
fi

DEV_BRANCH=true

# If building from a personal LBM fork, the script will store the built Docker
# image in a personal repository, as first-last-sagemaker-dev, instead of the
# main repository, lbm-sagemaker, where the production images are stored.

if [[ -z "${IMAGE_NAME}" ]]; then
  CUSTOM_IMAGE_NAME=false
  if ${DEV_BRANCH}; then
    email=$(git config user.email)
    username=$(echo ${email} | cut -d "@" -f 1)
    first_name=$(echo ${username} | cut -d "." -f 1)
    last_name=$(echo ${username} | cut -d "." -f 2)
    IMAGE_NAME=${first_name}-${last_name}-sagemaker-dev
  else
    IMAGE_NAME=variable-length-flow-policy
  fi
else
  CUSTOM_IMAGE_NAME=true
fi

# Confirm the configuration before the actual build process.
echo "Building sagemaker docker image with these settings:"
echo "  Fork Name:   ${FORK_NAME}"
echo "  Branch Name: ${BRANCH_NAME}"
echo "  AWS Region:  ${REGION}"
echo "  Ubuntu Version:  ${UBUNTU_VERSION}"
echo "  Image Name:  ${IMAGE_NAME}"
echo "  Repo Tag:  ${IMAGE_TAG}"
read -p "Are you sure this Docker configuration is correct? [Y/n]" RESPONSE
if [ "${RESPONSE}" != "y" ] && [ "${RESPONSE}" != "Y" ] && [ "${RESPONSE}" != "" ]; then
  echo "Exiting..."
  exit
fi

# Clean up a temp folder if exits from the previously failed build.
if [ -d tmp_docker_build ]; then
  rm -rf tmp_docker_build
fi
# Create a temp folder to build/push the Docker image.
mkdir tmp_docker_build && cd tmp_docker_build

# Clone the LBM repo either from the main branch of `robotics/lbm` or from a
# feature branch of a personal LBM fork.
echo "Cloning variable-length-flow-policy repo..."
echo "Building the docker image with ${FORK_NAME}/variable-length-flow-policy ${BRANCH_NAME}..."
git clone --quiet https://github.com/${FORK_NAME}/variable-length-flow-policy.git
cd variable-length-flow-policy && git checkout ${BRANCH_NAME} && git log -n 1 && cd ..
echo -e "\n"

# Prepare for `docker build`.
# (1) Copy necessary files into `tmp_docker_build`.
# (2) Modify Dockerfile based on the AWS region and take out torch from the
#     copied `requirements.in`.
# (3) Purge `../build.log` to start fresh.
# (4) Also, we need to log in to the public Amazon ECR to pull the base image
#     (specified in Dockerfile).
# 
# NOTE: If there is any change, please sync the same logic to 
# ./lbm_ecs_pipeline/docker/build_and_push_image.sh
echo "Building a Docker image named ${IMAGE_NAME}..."
cp variable-length-flow-policy/sagemaker/docker/Dockerfile .
cp variable-length-flow-policy/sagemaker/docker/.dockerignore .
sed -i -e "s/REGION_PLACEHOLDER/$REGION/g" Dockerfile
cp variable-length-flow-policy/${REQUIREMENTS_FILE} .
cp variable-length-flow-policy/conda_environment.yaml .        # <— NEW
sed -i "/^torch==/d" ${REQUIREMENTS_FILE}

AWSCLI_BIN=${lbm_dir}/tools/awscli/dist/aws

"${AWSCLI_BIN}" ecr get-login-password --region ${REGION}  | docker login --username AWS --password-stdin ${DOCKER_REGISTRY}
docker build --build-arg REQUIREMENTS_FILE=${REQUIREMENTS_FILE} --build-arg BASE_IMAGE=${BASE_IMAGE} --build-arg CONDA_ENV_NAME=robodiff-lh  . -t ${IMAGE_NAME}:${IMAGE_TAG} > build.log 2>&1

# Push the Docker image to our ECR. For that, we need to log in first.
account=$("${AWSCLI_BIN}" sts get-caller-identity --query Account --output text)
REPOSITORY_NAME="${account}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Pushing the SageMaker image to ECR repository: ${REPOSITORY_NAME}..."
# Suppress the error output if the repository already exists.
"${AWSCLI_BIN}" ecr create-repository --repository-name "${IMAGE_NAME}" --region ${REGION} > /dev/null 2>&1 || true

"${AWSCLI_BIN}" ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REPOSITORY_NAME}
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REPOSITORY_NAME}
docker push ${REPOSITORY_NAME}

# Print out a reminder if a development Docker image is built.
if ${DEV_BRANCH} || [ ${REGION} != "us-east-1" ]; then
  message="$ bazel run //lbm_sagemaker:launcher --"
  echo -e "\n"
  echo "Note: Building a non-standard Docker image."
  echo "Use the following command to submit SageMaker jobs:"
  if ${DEV_BRANCH} || ${CUSTOM_IMAGE_NAME}; then
    message="${message} --image-uri ${IMAGE_NAME}"
  fi

  if [ ${REGION} != "us-east-1" ]; then
    message="${message} --region ${REGION}"
  fi
  echo ${message}
  echo -e "\n"
fi

# Clean up resources to build fresh every time.
cd ..
mv tmp_docker_build/build.log .
rm -rf tmp_docker_build
echo "Build complete. Check build.log for the full log."
