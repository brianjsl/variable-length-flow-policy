# Training LBM on SageMaker

This README is mainly for training on SageMaker with the assumption that you
don't need or already have a Docker image. Regarding how to build a Docker
image, see [//lbm/setup/docker/README](./../setup/docker/README.md).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Submit SageMaker Jobs](#submit-sagemaker-jobs)
  - [Multitask Training](#multitask-training)
  - [Single-Task Training](#single-task-training)
  - [Specify AWS Region for the Training](#specify-aws-region-for-the-training)
- [Checkpoints and Artifacts from a SageMaker Job](#checkpoints-and-artifacts-from-a-sagemaker-job)
- [Hyperparameter Sweeps with W&B](#hyperparameter-sweeps-with-wb)
- [SageMaker Job Scheduling via AWS Batch](#sagemaker-job-scheduling-via-aws-batch)
  - [Understand Batch Sagemaker Job states](#understand-batch-sagemaker-job-states)
- [Monitor/Troubleshooting Submitted Jobs](#monitortroubleshooting-submitted-jobs)
  - [Sagey Tool](#sagey-tool)
  - [Batchy Tool](#batchy-tool)

## Prerequisites

Ensure you have gone through [first-time only README](../setup/first-time.md)
already to obtain AWS SageMaker setup.

Login with your AWS `sagemaker` profile.
```
$ cd <your-lbm-repo>
$ AWS_PROFILE=sagemaker ./aws sso login
```

## Submit SageMaker Jobs

All the SageMaker-specific configurations, e.g., the number of nodes and node
types, are specified in a launcher file. A launcher file also exposes some of
the frequently changed arguments to configure different types of training
before submitting a SageMaker job. For example, `config-file`, `batch-size`,
`num-epochs`, `task-name`, etc.

### Multitask Training

Run the launcher script to submit a training job. By default, this will use
the image stored in our main repository, `lbm-sagemaker`, for the training.
```
$ AWS_PROFILE=sagemaker bazel run //lbm_sagemaker:launcher
```

Run the launcher script for development and testing. Most importantly, specify
your development repository for the training.
```
$ AWS_PROFILE=sagemaker bazel run //lbm_sagemaker:launcher -- \
    --image-uri foo-bar-sagemaker-dev \
    --config-file A_TEST_CONFIG --num-epochs 1 --batch-size 50
```

Note that your W&B key and your email address will be automatically fetched from
`~/.netrc` and `$ git config user.email`, respectively.

If you want to manually provide the W&B key, do as follows:
```
$ export WANDB_API_KEY=<your-wandb-key>
```

Below is a sample terminal output indicating a SageMaker job is submitted:
```
sagemaker.config INFO - Not applying SDK defaults from location: /etc/xdg/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /home/kunimatsu/.config/sagemaker/config.yaml
INFO:sagemaker:Creating training-job with name: kunimatsu-lbm-multinode-training-2024-04-25-16-05-04-258
2024-04-25 16:05:05 Starting - Starting the training job...
2024-04-25 16:05:05 Pending - Training job waiting for capacity.........
2024-04-25 16:07:01 Pending - Preparing the instances for training..
```

### Specify AWS Region for the Training

We support SageMaker training in `us-east-1` and `us-west-2`. `us-east-1` is the
default if no region is specified. To run training on another region, you can
run your regular training command listed above with `--region us-west-2`
argument.

```
$ AWS_PROFILE=sagemaker bazel run //lbm_sagemaker:launcher -- \
    --image-uri foo-bar-sagemaker-dev \
    --config-file A_TEST_CONFIG --num-epochs 1 --batch-size 50 \
    --region us-west-2
```

However, you need to ensure your Docker image and training data is available in
`us-west-2`. When have doubts, post a question in #lbm-policy-software channel.


### Single Task Training

This feature has been removed in [PR#1607](https://github.shared-services.aws.tri.global/robotics/lbm/pull/1607)


## Checkpoints and Artifacts from a SageMaker Job

### S3 Bucket for Different AWS Regions

All the SageMaker artifacts will be stored in this S3 bucket
`s3://robotics-manip-lbm/sagemaker_outputs/` for training in the `us-east-1`
region. For training in the `us-west-2` region, the artifacts will be stored in
`s3://robotics-manip-lbm-us-west-2/sagemaker_outputs/` instead.

### S3 Subdirectory for SageMaker Runs

To better manage the artifacts generated from different ECR repositories and
runs, we further extend S3 paths based on the code being run, i.e., the LBM main
branch or a user's dev branch, and whether it's single-task or multi-task.

This is an example for the `us-east-1` S3 directory structure:
```
s3://robotics-manip-lbm/sagemaker_outputs
├── main-lbm
│   └── main-lbm-training-YYYY-MM-DD-HH-MM-SS/
│   └── main-lbm-training-YYYY-MM-DD-HH-MM-SS/
├── main-single-task
│   └── main-{task_one_name}-training-YYYY-MM-DD-HH-MM-SS/
│   └── main-{task_two_name}-training-YYYY-MM-DD-HH-MM-SS/
├── usera-lbm
│   └── usera-lbm-training-YYYY-MM-DD-HH-MM-SS/
│   └── usera-lbm-training-YYYY-MM-DD-HH-MM-SS/
├── usera-single-task
│   └── usera-{task_one_name}-training-YYYY-MM-DD-HH-MM-SS/
│   └── usera-{task_two_name}-training-YYYY-MM-DD-HH-MM-SS/
```

### SageMaker Checkpoints

Checkpoints are the most important artifacts from a SageMaker run, and they will
be synced to S3 periodically given the config settings (`training.val_every`
and `checkpoint.every_n_steps`) of the run.

Here is an example command to download a checkpoint to local:
```
$ cd <your-lbm-dir>
$ AWS_PROFILE=manip-cluster ./aws s3 cp s3://robotics-manip-lbm/sagemaker_outputs/main-lbm/main-lbm-training-YYYY-MM-DD-HH-MM-SS/epoch_X_-step-Z.ckpt .
```

An example command to show a list of checkpoints:
```
$ cd <your-lbm-dir>
$ AWS_PROFILE=manip-cluster ./aws s3 ls s3://robotics-manip-lbm/sagemaker_outputs/main-lbm/main-lbm-training-YYYY-MM-DD-HH-MM-SS/ --recursive
```

### Fully Resolved Config for a SageMaker Run

For the fully resolved config of the run, you can find it in the associated W&B
page under `Files/config.yaml`. **Note that a W&B run and a SageMaker run share
the same name**, so you should be able to locate them from one to another.

You can also find W&B logs and files in your S3 folder. They will only be synced
at the end of a SageMaker run (whether finished or terminated). You can find
them in a tar file under `S3_DIR_OF_YOUR_RUN/output/output.tar.gz`.

### View SageMaker logs

> [!NOTE]
>
> **Key Change in Workflow: Retrieving SageMaker Training Logs**
>
> Previously, submitting jobs directly to SageMaker allowed you to view real-time logs in the terminal, including pending capacity updates and runtime logs.
>
> With the transition to AWS Batch, real-time log streaming is no longer available. Instead, you will need to use the `sagey logs <job-name>` command to retrieve the logs.
>
> The LBM launcher submits the job to the queue and monitors its status until it reaches the "Running" state. Once the job is running, you can use the `sagey` CLI tool to fetch runtime logs and view logs for completed jobs.
>
> **Slack Integration**
>
> IE added Slack integration to notify users when a SageMaker job transitions to the "RUNNING" state to provide better visibility into job execution. Notifications also include updates for "FAILED" states. You may want to join the [#robotics-sagemaker-job-alerts](https://tri.enterprise.slack.com/archives/C07QHCYR2UF) slack channel to easily receive these notices.


You can use the sagey logs command to view the logs in the terminal:

`sagey logs <job-name>`: Tails the live logs for your job. Pass the `--startfromhead` option to retrieve logs from the beginning.

You can find more info on sagey here [sagey-tool](#sagey-tool).

### Terminate a SageMaker Job

**Canceling a Batch Job from the Queue**

If the job is in the RUNNABLE or STARTING state (i.e., waiting for execution in
the queue), you can remove it from the queue using the following command:
```
$ batchy cancel <QUEUE-NAME> <job-name>
```
You can find more info on batchy here [batchy-tool](#batchy-tool).

**Stopping a Running Job**
If the job is already under execution, you can terminate it in one of the
following ways:

1. Via SageMaker Dashboard:
  For `us-east-1`, go to the [SageMaker Dashboard](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/dashboard)
  and terminate the jobs. For `us-west-2`, use the [SageMaker Dashboard](https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs)
  to terminate the jobs.
2. Using sagey CLI. You can find more info on sagey here
[sagey-tool](#sagey-tool). Alternatively, you can use `sagey` to kill the job.
  ```
  $ sagey stop job1-name --region <AWS-REGION>
  ```

## Hyperparameter Sweeps with W&B

`sweeper.py` launches SageMaker jobs for hyperparameter sweeps using W&B.

### Usage

See the help for the arguments:

```
$ AWS_PROFILE=sagemaker bazel run //lbm_sagemaker:sweeper -- --help
```

#### Agents and Runs

Two of the available arguments are `--max-agents` and `--max-runs`.

"Agents" is a W&B Sweep term. An Agent manages launching runs and selecting new
parameter values based on the logged metrics by previous runs.

The `--max-agents` command allows you to use multiple agents to manage your
sweep, allowing multiple parallel runs of a sweep to happen. For example,
`--max-agents=2` means two sagemaker training jobs are launched at once, each
logging metrics to the overall sweep. Each agent would launch new jobs as old
ones complete, until the parameter(s) are effectively swept. On the other hand,
setting `--max-agents=1` would launch one sagemaker training job, wait for it to
complete, and launch the next.

The `--max-runs` command allows you to limit the total number of runs amongst
all agents. This value should be divisiable by the number of agents. For
example, lets say you wanted to sweep a parameter that may require 20 different
values tested by an agent. If you wanted to run a small sweep of only 6 runs
total, you could set `--max-runs=6`

#### Example Command

```
$ AWS_PROFILE=sagemaker bazel run //lbm_sagemaker:sweeper -- \
    --config-file train_diffusion_unet_clip_drake_single_task_rel_prop \
    --num-epochs 5 --instance-type p4d --batch-size 18 \
    --config-overrides training.val_per_step=True,training.val_every=5 \
    --sweep-config-file example_sweep.yaml
```

### Configuration Files

#### Training Configuration

The training configuration file should be placed in the
`diffusion_policy/config` folder, and should follow the standard hydra format.
The default file is `train_diffusion_unet_clip_lang_stage_base.yaml`.

#### Sweep Configuration - Basic Example

The sweep configuration file should be placed in the `./sweep/` directory. The
default file is `example_sweep.yaml`.

An basic example might look like:
```yaml
project: diffusion_sweep
entity: tri

method: bayes #options: grid, random, bayes
metric:
  goal: minimize
  name: val_loss
parameters:
  optimizer:
    parameters: #nested parameters need to have "parameters" key at each level
      lr:
        min: 0.00001
        max: 0.001
```
A few notes of the `parameter` key:
- What this shows is that `optimizer.lr` will be sweeped with a minimum value of
`1e-5` and max value of `1e-3`, selected via bayesian sampling.
- `optimizer.lr` follows where learning rate is stored in the hydra config
structure. **The sweep parameters need to follow the hydra structure so that
they can properly overwrite the hydra values.**
- Per the [W&B Documentation](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#double-nested-parameters),
nested parameters (such as `optimizer.lr`) need to have the `parameters` key
between each level.

You can see the [sweep basic structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#double-nested-parameters)
and [configuration options](https://docs.wandb.ai/guides/sweeps/sweep-config-keys)
guides from W&B for all config options and more info.

#### Sweep Configuration - Conditional Sweeps

You may want to sweep something that requires multiple changes to a hydra
configuration, dependent on the value selected by the sweep agent. An example
could be sweeping the backbone. Below shows a potential example of sweeping a
models backbone between CLIP and resnet18:

```yaml
project: diffusion_sweep
entity: tri

method: grid
metric:
  goal: minimize
  name: val_loss
parameters:
  conditional: # Use the "conditional" key to look for the config file in the "configs" directory
    values: ["example_clip_addon", "example_resnet"]
    distribution: categorical
```

The parameter swept is `conditional`, and it is a categorical value between
`example_clip_addon`, `example_resnet`. `conditional` is a special sweep
parameter that looks for a config in `./sweep/config/conditional`. For example,
below is `./sweep/config/conditional/example_clip_addon.yaml`:

```yaml
policy:
  obs_encoder:
    rgb_model:
      _target_: diffusion_policy.model.vision.model_getter.get_clip_vit_base_patch16
    resize_shape: [256, 342]
    crop_shape: [224, 224]
    share_rgb_model: True

training:
  lr_scale_for_grouped_params:
    obs_encoder: 0.1
```
and `./sweep/config/conditional/example_resnet.yaml`:
```yaml
policy:
  obs_encoder:
    rgb_model:
      _target_: diffusion_policy.model.vision.model_getter.get_resnet
      name: resnet18
      weights: IMAGENET1K_V1
```
This structure allows you to set as many conditional values you need to change
from a config file. This config overwrites the final hydra config, so **the
structure should follow the structure of your hydra config file.**

#### Sweeping Multiple Parameters at Once

Multiple parameters can be swept at once, and follows a similar format to the
examples above. Please see a [combined example](./sweep/config/example_sweep_combined.yaml)
for reference.

Note - when combining a conditional (i.e. categorical) paramenter with a
continuous parameter in sweeps, you should review the
[sweep methods](https://docs.wandb.ai/guides/sweeps/sweep-config-keys#method)
and how each method will affect the sweep agent's selection process for each
parameter.

## SageMaker Job Scheduling via AWS Batch

We are transitioning to AWS Batch to manage SageMaker Training Jobs to enhance
job queuing and resource management. All SageMaker jobs utilizing p4, p4de, and
p5 instances in *us-east-1* and *us-west-2* regions will be submitted through
AWS Batch. This approach allows training jobs to be routed to an AWS Batch queue
rather than directly to SageMaker, enabling batch processing, efficient resource
allocation, and better scalability.

By adopting AWS Batch, we aim to streamline training job management and maximize
resource usage efficiency across regions.


### How It Works

The launcher script submits SageMaker jobs to designated AWS Batch queues. These
queues are preconfigured by the IE team, with each queue corresponding to a
specific instance type per region. For example, here’s an outline of queue
configurations:

**Region-specific Queues:**

To optimize resource allocation, each AWS Batch queue is tailored to a specific
instance type (e.g., p4, p4de, p5) in designated regions. Below is a reference
for the queue names per instance type:

**us-west-2 Region:**
- ml.p5.48xlarge: fss-ml-p5-48xlarge-us-west-2
- ml.p4de.24xlarge: fss-ml-p4de-24xlarge-us-west-2
- ml.p4d.24xlarge: fss-ml-p4d-24xlarge-us-west-2

You can also use the following command to list all queues for a specific region:
```
$ batchy list-queues --region us-west-2
```

## Understand Batch Sagemaker Job states

Once a job is submitted via the launcher, it enters the SUBMITTED state. AWS
Batch will then allocate resources and move the job through its lifecycle
states.

> [!NOTE]
> Quick overview of what different queue job states mean:
> 1. **SUBMITTED:** The job has been submitted and is awaiting execution.
> 2. **PENDING:** The job is waiting for resources or other conditions to be met
before execution.
> 3. **RUNNABLE:** The job is ready to be run but has not yet started.
> 4. **STARTING:** The job is in the process of starting but has not yet fully
begun.
> 5. **SCHEDULED:** The job is in the queue and waiting to be scheduled for
execution.
> 6. **SUCCEEDED:**  A job has been completed successfully.
> 7. **FAILED:**  The job encountered an error and did not complete
successfully.

## Resource Allocation for Prototyping

AWS Batch operates with a fair share policy to ensure an equitable distribution of resources over time, but this can introduce latency. Unfortunately, instances cannot be reserved per team within the batch queue. If long-running jobs occupy all available nodes, it may hinder rapid prototyping.

To address this, separate queues have been created for sensitive prototyping, starting with a limited instance count.

### Teams with Dedicated Prototyping Queues:
- **MVDP**
- **VLA**
- **DP**
- **DAS**

To ensure the correct queue is selected, pass the `--team` parameter when running the launcher file. This directs jobs to the appropriate prototyping queue based on the specified team.

## Monitor/Troubleshooting Submitted Jobs

To monitor job progress and troubleshoot issues, you can use the following
command-line tools:

- **sagey**: Useful for interacting with SageMaker training jobs.
- **batchy**: Designed for querying AWS Batch queues and understanding job
statuses.

### Sagey Tool

There is a command-line tool for SageMaker called
[`sagey`](https://github.com/TRI-ML/sagey). It has several useful features. Ping
IE if you don't have access permission.

> [!INFO]
> `sagey` is installed during `./setup/install_prereqs.sh` script so this makes it available in venv by default.

To activate the venv, run this:

```sh
# Activate the venv.
$ source <lbm>/activate-venv.sh
```

Here are some useful commands:
- `sagey ls`: List the jobs that are currently running or recently submitted,
  along with their status.

- `sagey usage`: List the number of used and pending jobs as well as quota
  for each instance type.

- `sagey stop <job-name>`: Kill the submitted job that has `<job-name>`.
  You can look for your `<job-name>` via `sagey ls`.

- `sagey logs <job-name>`: Tails the live logs for your job. Pass the `--startfromhead` option to retrieve logs from the beginning.

  You can use that command to find the link to your `wandb` run:
  ```
  $ AWS_PROFILE=sagemaker sagey logs <job-name> | grep "wandb.ai"
  ```

If the job is downloaded, the logs are automatically downloaded locally.
  ```
  $ AWS_PROFILE=sagemaker sagey logs <job-name>
  ```

> [!NOTE]
>
> If you want to run `sagey` against `us-west-2`, prepend
`AWS_DEFAULT_REGION=us-west-2` in front of the sagey command or pass --region
option to the end of the command like this `--region us-west-2`.
>
> You can also pass the profile option to the command if you don't want to set `AWS_PROFILE`.
`sagey ls --profile sagemaker --region us-west-2`

### Batchy Tool

You can install batchy as follows:

> [!INFO]
> `batchy` is installed during `./setup/install_prereqs.sh` script so this makes it available in venv by default.

To activate the venv, run this:

```sh
# Activate the venv.
$ source <lbm>/activate-venv.sh
```

This tool has several commands to help monitor batch queues. See the
command-line help `batchy --help` for more details.

  - This command will list all the queues in the AWS Batch:
  ```
  $ batchy list-queues --region us-west-2
  ```

  - Display the Status of Running and 'In Queue' Jobs in a queue:
  ```
  $ batchy ls fss-ml-p4de-24xlarge-us-west-2 --region us-west-2 --profile sagemaker
  ```

  - This command will print the model metrics for a given job:
  ```
  $ batchy model-metrics <QUEUE-NAME> <JOB-NAME>
  ```

  - Display the Completed Job in the queue:
  ```
  $ batchy completed-jobs fss-ml-p4de-24xlarge-us-west-2 --region us-west-2 --profile sagemaker
  ```

  - Cancel a Job in the Queue:
  ```
  $ batchy cancel-job <QUEUE-NAME> <JOB-NAME>
  ```

  - You can add nicknames to queues so that we don't have to put the full name each time:
  `
  $ batchy add-queue nickname queue-name
  `
  ```
  $ batchy ls fss-ml-p5-48xlarge-us-east-1 # -> shows me the output correctly
  $ batchy add-nickname p5 fss-ml-p5-48xlarge-us-east-1
  $ batchy ls p5
  ```  $ batchy add-queue nickname queue-name
  `
  ```
  $ batchy ls fss-ml-p5-48xlarge-us-east-1 # -> shows me the output correctly
  $ batchy add-nickname p5 fss-ml-p5-48xlarge-us-east-1
  $ batchy ls p5
  ```
