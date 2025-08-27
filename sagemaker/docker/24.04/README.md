# Ubuntu 24.04 PyTorch SageMaker Docker Image

This folder contains a Dockerfile and a script to build and push a Docker image for PyTorch SageMaker on Ubuntu 24.04. This custom image is stored at `124224456861.dkr.ecr.us-west-2.amazonaws.com/lbm-pytorch-training:2.6.0-gpu-py312-cu126-ubuntu24.04-sagemaker`.

## Dockerfile

The Dockerfile in this repository is used to create the custom Docker image for PyTorch SageMaker on Ubuntu 24.04. This Dockerfile is the modified version of the one for the latest [AWS PyTorch SageMaker base images](https://github.com/aws/deep-learning-containers/blob/351c7f9a636fdc536ef2ee708f92bfbea4319548/pytorch/training/docker/2.5/py3/cu124/Dockerfile.gpu) (`pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker`).

## Build SageMaker Images and Upload to Amazon ECR

Unless you need to make changes to the underlying base image, you do not need to build/upload the image. If you do need to make changes to the base image, we wrap all the steps of building and uploading a SageMaker Docker image into a
convenient [script](./build_and_push_base_image.sh).

```
$ cd <path/to/your/lbm>/setup/docker/24.04
$ AWS_PROFILE=sagemaker ./build_and_push_base_image.sh
```

## Usage

To use this base image for LBM training, you can simply use the scripts described in the main [docker README](../README.md)

```
$ cd <path/to/your/lbm>/setup/docker
$ AWS_PROFILE=sagemaker ./build_and_push_image.sh UBUNTU_VERSION=24.04
```
