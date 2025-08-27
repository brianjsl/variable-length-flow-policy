REGION=us-west-2
IMAGE_NAME=lbm-pytorch-training
IMAGE_TAG=2.6.0-gpu-py312-cu126-ubuntu24.04-sagemaker
AWS_DL_REPO=https://raw.githubusercontent.com/aws/deep-learning-containers/v1.6-pt-ec2-2.5.1-tr-py311
AWSCLI_BIN=$(dirname $0)/../../../tools/awscli/dist/aws

set -e

for named_arg in "$@"
do
  KEY=$(echo ${named_arg} | cut -f1 -d=)
  if [ $KEY != "REGION" ]; then
    echo "Only REGION (case sensitive) is allowed as an argument."
    exit
  fi
  KEY_LENGTH=${#KEY}
  VALUE="${named_arg:$KEY_LENGTH+1}"
  declare "$KEY"="$VALUE"
done

# Clean up a temp folder if exits from the previously failed build.
if [ -d tmp_docker_build ]; then
  rm -rf tmp_docker_build
fi
# Create a temp folder to build/push the Docker image.
mkdir tmp_docker_build

# Download necessary files from the AWS Deep Learning Containers repository.
echo "Downloading necessary files from the AWS Deep Learning Containers repository..."

FILES=(
    "pytorch/training/docker/build_artifacts/start_cuda_compat.sh"
    "pytorch/training/docker/build_artifacts/dockerd_entrypoint.sh"
    "pytorch/training/docker/build_artifacts/changehostname.c"
    "pytorch/training/docker/build_artifacts/start_with_right_hostname.sh"
    "src/deep_learning_container.py"
)

LOG_FILE="tmp_docker_build/downloads.log"

for FILE in "${FILES[@]}"; do
    wget "${AWS_DL_REPO}/${FILE}" -O "tmp_docker_build/$(basename ${FILE})" -a "${LOG_FILE}"
done


account=$("${AWSCLI_BIN}" sts get-caller-identity --query Account --output text)
REPOSITORY_NAME="${account}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

"${AWSCLI_BIN}" ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${REPOSITORY_NAME}


docker build . -t ${IMAGE_NAME}:${IMAGE_TAG} 
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REPOSITORY_NAME}:${IMAGE_TAG}
docker push ${REPOSITORY_NAME}:${IMAGE_TAG}

rm -rf tmp_docker_build
