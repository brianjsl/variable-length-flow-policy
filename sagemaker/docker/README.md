# LBM Docker Images

Note: If you only want to try out training on SageMaker, depending on when the
latest Docker image is built, you don't necessarily need to build a new Docker
image. If that's the case, skip to [Training LBM on SageMaker](../../lbm_sagemaker/README.md)
for instructions.

LBM Docker workflow is mainly leveraged to run distributed training on
[Amazon SageMaker](https://aws.amazon.com/sagemaker/) platform. We don't
typically develop our code in a Docker environment.

## Prerequisites

Ensure you have gone through [first-time only README](../first-time.md) already
to obtain AWS SageMaker setup.

### Install Docker

Follow [this Nvidia page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.13.5/install-guide.html#docker)
to install Docker and `nvidia-container-toolkit`.

Run this command to ensure that you can run Docker commands without `sudo`:
```
sudo usermod -aG docker $USER
```
Note that if you are connected via `ssh` or using `tmux`, you may need to log
out and log back in and/or restart `tmux` in order for the changes to take
effect.

Confirm that you can run Docker commands without `sudo`:
```
docker run hello-world
```

Run this command to ensure the Docker container can access GPUs.
```
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

### Verify your Credentials

You should see `Robotics-LBM-PowerUserAccess` under the `ml-sandbox-16011`
account via the [landing page](https://tri-sso.awsapps.com/start#/).

Note that SSO credentials expires every 3 days, so you need to log back in
periodically. On the contrary, `instance roles`, e.g., ` dev-ecs-instance-role`,
used for EC2 instances don't expire.

On a terminal, login with your TRI and Amazon ECR credentials so that you can
pull base Docker images from Amazon ECR. Remember to replace `us-east-1` to the
region, e.g., `us-west-2`, you intend to run SageMaker on.
```
$ AWS_PROFILE=sagemaker ./aws sso login
$ AWS_PROFILE=sagemaker ./aws ecr get-login-password --region us-east-1  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Below is the sample terminal output:
$ AWS_PROFILE=sagemaker ./aws sso login
Attempting to automatically open the SSO authorization page in your default browser.
If the browser does not open or you wish to use a different device to authorize this request, open the following URL:

https://device.sso.us-east-1.amazonaws.com/

Then enter the code:

WMGH-TMNJ

$ AWS_PROFILE=sagemaker ./aws ecr get-login-password --region us-east-1  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
WARNING! Your password will be stored unencrypted in /home/zachfang/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

## Build SageMaker Images and Upload to Amazon ECR

We wrap all the steps of building and uploading a SageMaker Docker image into a
convenient [script](./build_and_push_image.sh).

**There are six named arguments (case-sensitive) for the script**:
- `FORK_NAME`: LBM fork to clone. Default to the main repository, `robotics`.
- `BRANCH_NAME`: Branch to checkout the code. Default to `main`.
- `REGION`: Which AWS region to push the build image. Only `us-east-1` and
  `us-west-2` are supported. Default to `us-east-1`.
- `UBUNTU_VERSION`: Which sagemaker base image to use, determined by the ubuntu version. Default to `20.04`
- `IMAGE_NAME`(Optional): Image name. If not specified (default), it resolves
  to `lbm-sagemaker` in the case of the LBM main or
  `{first}-{last}-sagemaker-dev` otherwise.
- `IMAGE_TAG`(Optional): Image tag name. The default is `latest`.
```
# Build and push the Docker image with default settings. Normally, this should
# take a few minutes. You can check the terminal output or `build.log` if there
# is any unexpected error.
$ cd setup/docker
$ AWS_PROFILE=sagemaker ./build_and_push_image.sh

# Build and push images to `us-west-2`.
$ AWS_PROFILE=sagemaker ./build_and_push_image.sh REGION=us-west-2
```

Note: This step is **only** for building images from the main branch and pushing
them to the main ECR repository in `us-east-1`, [lbm-sagemaker repository](https://us-east-1.console.aws.amazon.com/ecr/repositories/private/124224456861/lbm-sagemaker?region=us-east-1).
This repository should only store production images for the actual training
images. See the section below for the development workflow.

Below is a sample terminal output for a successul image build:
```
$ AWS_PROFILE=sagemaker ./build_and_push_image.sh
Building sagemaker docker image with these settings:
  Fork Name:   robotics
  Branch Name: main
  AWS Region:  us-east-1
  Ubuntu Version: 20.04
Are you sure this Docker configuration is correct? [Y/n]y
Cloning LBM repo...
Building the docker image with robotics/lbm main...
Already on 'main'
Your branch is up to date with 'origin/main'.
commit ec3c48defe79daa27cde619e8d0c039e8ca1f5bc (HEAD -> main, origin/main, origin/HEAD)
...

Building a Docker image named lbm-sagemaker...
WARNING! Your password will be stored unencrypted in /home/zachfang/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
Pushing the SageMaker image to ECR repository: 124224456861.dkr.ecr.us-east-1.amazonaws.com/lbm-sagemaker:latest...
WARNING! Your password will be stored unencrypted in /home/zachfang/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
The push refers to repository [124224456861.dkr.ecr.us-east-1.amazonaws.com/lbm-sagemaker]
682cacc461d5: Pushed
78fda418a7c1: Layer already exists
...
latest: digest: sha256:d165778099441cf685e1788803d533188abe14e23f0486d75159d050fa8fc368 size: 8746
Build complete. Check sagemaker_base/build.log for the full log.
```

### Create and Train with Development SageMaker Images

If you are developing or testing some features to run on SageMaker, you can
build a Docker image using the code in a branch.

Note: To promote traceability, we don't build a Docker image with your local
changes. They **need to be pushed to a branch in GHE**.

To do so, specify **FORK_NAME** and **BRANCH_NAME** from the LBM fork of the
specified user when building your SageMaker images. By providing those
info, the script will:
- Checkout the specific feature branch.
- Create a personal repository automatically, named `first-last-sagemaker-dev`,
  on Amazon ECR if it doesn't exist.
- Push the built images to it.

```
# The script will checkout that feature branch and copy the code into the Docker
# image.
$ cd setup/docker
$ AWS_PROFILE=sagemaker ./build_and_push_image.sh FORK_NAME=GHE_USERNAME BRANCH_NAME=BRANCH_NAME
```

After you have built and uploaded the image from your terminal, verify it again
on [the Amazon ECR page](https://us-east-1.console.aws.amazon.com/ecr/private-registry/repositories?region=us-east-1).
Ask IE, or Satya Kotari specifically, to grant you the pull permission for the
repository.

Below is a sample terminal output for a successful image build:
```
$ AWS_PROFILE=sagemaker ./build_and_push_image.sh FORK_NAME=zach-fang BRANCH_NAME=feature/test-branch
Building sagemaker docker image with these settings:
  Fork Name:   zach-fang
  Branch Name: feature/test-branch
  AWS Region:  us-east-1
  Ubuntu Version: 20.04
Are you sure this Docker configuration is correct? [Y/n]y
Cloning LBM repo...
Building the docker image with zach-fang/lbm feature/test-branch...
Branch 'feature/test-branch' set up to track remote branch 'feature/test-branch' from 'origin'.
Switched to a new branch 'feature/test-branch'
commit a9b7f4c6dcf18380d9ffd0c858bc9b429f7b255c (HEAD -> feature/test-branch, origin/feature/test-branch)
...

Building a Docker image named zach-fang-sagemaker-dev...
WARNING! Your password will be stored unencrypted in /home/zachfang/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store


Login Succeeded
Pushing the SageMaker image to ECR repository: 124224456861.dkr.ecr.us-east-1.amazonaws.com/zach-fang-sagemaker-dev:latest...
WARNING! Your password will be stored unencrypted in /home/zachfang/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
The push refers to repository [124224456861.dkr.ecr.us-east-1.amazonaws.com/zach-fang-sagemaker-dev]
26d131b59be2: Preparing
...
latest: digest: sha256:622494a5a95dafc2b9152eb1cf4957892f36eb8cb5ab5fe9b2c665e4c1090532 size: 9579


Note: Building a development Docker image.
Use the following command to submit SageMaker jobs:
$ bazel run //lbm_sagemaker:launcher -- --image-uri zach-fang-sagemaker-dev


Build complete. Check sagemaker_base/build.log for the full log.
```

### Submit SageMaker Jobs

Once you have built a Docker image, you can proceed to
[Training LBM on SageMaker](./../../lbm_sagemaker/README.md) to submit a
SageMaker training job.


## (Optional) SageMaker Quick Start

If you want to know more about the SageMaker setup, there is a great
documentation [AWS SageMaker Quick Start](https://docs.google.com/document/d/1cpbls8qyP3k0GoaDXEJcAYHtvFFJU6mLGgOIXIk04n4/edit#heading=h.e7jlgmlwg6wr)
from ML team's Dian Chen.


## Building the docker container and training locally

The docker container has all the dependencies and permissions for building and
training locally (i.e. on your Puget). To build the image, run the following:

```
$ ./setup/docker/build_and_run_local.sh build
```

This will copy your local `lbm` code including non-commited local changes into
the docker container. Once the container has been built, run it with the
following command:

```
$ ./setup/docker/build_and_run_local.sh run
```

The following message will show up on the console. This can be ignored:

```
wandb: Couldn't detect image argument, running command without the WANDB_DOCKER env variable
```

**Note:** the command above also passes WandB credentials as well as sets docker
properties for multi-node training (to be tested). ``

Finally, you can train a model. For a simple DP training run, execute:
```
$ cd lbm
$ AWS_PROFILE=manip-cluster python diffusion_policy/scripts/train.py --config-name train_diffusion_unet_resnet_datapipe_single_task.yaml
```
