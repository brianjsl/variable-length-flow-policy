"""Contains common utility functions/variables for SageMaker launchers."""

from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Dict

import boto3
from sagemaker.batch_queueing.boto_client import get_batch_boto_client
from sagemaker.batch_queueing.queue import Queue

# These are the TRI SageMaker account settings.
ML_ACCOUNT_ID = "124224456861"

ALLOWED_INSTANCE_TYPES = dict(
    p5="ml.p5.48xlarge",
    p4de="ml.p4de.24xlarge",
)

TASK_NAME_PADDING = "*padding*"

QUEUE_MAPPER = {
    "us-west-2": {
        "ml.p4de.24xlarge": {
            "default": "fss-ml-p4de-24xlarge-us-west-2",
            "mvdp": "fss-mvdp-p4de-24xlarge-us-west-2",
            "vla": "fss-vla-p4de-24xlarge-us-west-2",
            "das": "fss-das-p4de-24xlarge-us-west-2",
            "bd": "fss-lbm-tri-bd-p4de-24xlarge-us-west-2",
        },
        "ml.p5.48xlarge": {
            "default": "fss-ml-p5-48xlarge-us-west-2",
            "testing": "fss-testing-p5-48xlarge-us-west-2",
            "bd1": "fss-tri-bd-p5en-48xlarge-us-west-2",
            "bd2": "fss-tri-bd-p5-48xlarge-us-west-2",
        },
    },
}


def get_sagemaker_profile() -> str:
    client = boto3.client("sts")
    response = client.get_caller_identity()
    arn = response.get("Arn")
    return arn


def assert_sagemaker_profile():
    client = boto3.client("sts")
    response = client.get_caller_identity()
    arn = response.get("Arn")
    assert ML_ACCOUNT_ID in arn, (
        f"Found the wrong role: {arn}. Make sure to run your command with "
        "AWS_PROFILE=sagemaker"
    )


def assert_us_west_2_region(region):
    assert (
        region == "us-west-2"
    ), f"Using the wrong region: {region}. Only us-west-2 is allowed."


def assert_training_job_runtime(max_time_seconds):
    assert max_time_seconds <= 2419200, (
        f"The maximum allowed sagemaker training runtime is "
        f"28 days(2419200 seconds)."
        f"Requested {max_time_seconds} seconds."
    )


def maybe_set_wandb_group_name(overrides: Dict, job_name: str) -> Dict:
    """Sets `+logging.group` attribute if `overrides` doesn't have it yet.
    This is added to put them together the runs from the original one and
    the ones launched with `+training.resume_checkpoint_path`.
    """
    found = False
    for key, value in overrides.items():
        if key.endswith("logging.group"):
            found = True
            break

    if not found:
        overrides["+logging.group"] = job_name
    return overrides


def maybe_add_task_name_padding(task_name):
    """Adds a padding, `*padding*`, to a task name if its length is exactly
    forty. SageMaker performs a security check and considers any hyperparameter
    matching the below pattern to be unsafe.
    """
    # Note(zachfang): The false positive lint error couldn't be suppressed in a
    # multi-line comment.
    # SageMaker checking pattern: [A-Za-z0-9/\+=]{40}  # noqa W605
    if len(task_name) == 40:
        task_name = TASK_NAME_PADDING + task_name
    return task_name


def maybe_remove_task_name_padding(task_name):
    """Removes the padding if it exists. This is a reverse action of
    maybe_add_task_name_padding().
    """
    if task_name.startswith(TASK_NAME_PADDING):
        task_name = task_name[len(TASK_NAME_PADDING) :]
    return task_name


def get_wandb_api_key():
    """Gets wandb api key either from ~/.netrc or env variable."""
    netrc = Path.home().resolve() / ".netrc"
    if netrc.is_file():
        wandb_api_key = None
        with open(netrc, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.lstrip()
            if line.startswith("password"):
                wandb_api_key = line.split(" ")[-1].rstrip()
    else:
        wandb_api_key = os.environ.get("WANDB_API_KEY", None)
    assert wandb_api_key, "No wandb api key provided."
    return wandb_api_key


def get_hf_token_key():
    """Gets HuggingFace token either from ~/.hf_token or env variable. Note
    that we expect `.hf_token` to be a one-liner containing only the key.
    """
    hf_token = Path.home().resolve() / ".hf_token"
    if hf_token.is_file():
        hf_token_key = None
        with open(hf_token, "r") as f:
            lines = f.readlines()
        assert (
            len(lines) == 1
        ), "~/.hf_token should be a one-liner containing only the key."
        hf_token_key = lines[0].strip()
    else:
        hf_token_key = os.environ.get("HF_TOKEN", None)

    assert hf_token_key, (
        "No HuggingFace token provided. Make sure to go through the "
        "'HuggingFace Credential' section in `setup/first-time.md`."
        "The token can be specified via `HF_TOKEN` env variable or "
        "in `~/.hf_token`."
    )
    return hf_token_key


def get_email():
    """Gets the user email from the git setup."""
    output = subprocess.run(
        ["git", "config", "user.email"],
        capture_output=True,
        text=True,
    )
    assert output.returncode == 0, "Can't fetch the user email from git."

    email = output.stdout.rstrip()
    assert email.endswith("@tri.global")
    return email


def get_sagemaker_tags():
    """Gets the required SageMaker_tags for the training run."""
    sagemaker_tags = [
        {"Key": "tri.project", "Value": "LBM:PJ-0109"},
        {
            "Key": "tri.owner.email",
            "Value": get_email(),
        },
    ]
    return sagemaker_tags


def get_kms_encryption_key(*, region):
    """Returns the encryption key for syncing data between S3 and SageMaker.
    The key is dedicated for each AWS region and created by IE.
    """
    if region == "us-east-1":
        return "46af3d92-4e65-4a8e-b7ff-f8076d31d28d"
    elif region == "us-west-2":
        return "c66e8bb6-7dbe-497b-bc41-fb347d846548"
    else:
        raise RuntimeError(f"Unrecognized AWS region: {region}")


def get_s3_output_path(*, region, is_main_branch):
    """Gets the S3 output path based on the AWS region, whether it's running
    with the main branch, and if it's a single-task training.
    """
    # We append the region name for our `us-west-2` data bucket.
    bucket_suffix = "" if region == "us-east-1" else "-us-west-2"
    base_s3_outout_path = (
        f"s3://robotics-manip-lbm{bucket_suffix}/sagemaker_outputs"
    )
    s3_subdir = "main" if is_main_branch else os.getlogin()
    s3_subdir += "-variable-length-flow-policy"
    return f"{base_s3_outout_path}/{s3_subdir}"


def get_job_names(*, is_main_branch, task_name):
    """Returns `base_job_name` and `job_name` for the PyTorch estimator. We
    take full control of SageMaker job names to enforce SageMaker to save each
    run's output to exactly one folder.
    """
    # SageMaker enforces the full job name NOT longer than 63.
    JOB_NAME_LENGTH_LIMIT = 63

    job_prefix = "main" if is_main_branch else os.getlogin()
    launch_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if task_name:
        # Convert `CamelCase` to `snake-case-with-hyphens`.
        # fmt: off
        snake_case_with_hyphens = (
            re.sub("([A-Z]+)", r"-\1", task_name)[1:].lower()
        )
        # fmt: on

        # Find out the maximum length `job_prefix` can take.
        job_prefix_length = (
            JOB_NAME_LENGTH_LIMIT
            - len(job_prefix)
            - len(launch_time)
            - len("training")
            # Three hyphens in between four subcomponents of `full_job_name`.
            - 3
        )

        # Truncate `job_prefix` and remove duplicated hyphen in the end of the
        # string if needed.
        if len(snake_case_with_hyphens) > job_prefix_length:
            snake_case_with_hyphens = snake_case_with_hyphens[
                0:job_prefix_length
            ]
            if snake_case_with_hyphens.endswith("-"):
                snake_case_with_hyphens = snake_case_with_hyphens[:-1]
        job_prefix += f"-{snake_case_with_hyphens}"
    else:
        job_prefix += "-lbm"

    base_job_name = f"{job_prefix}-training"
    full_job_name = f"{base_job_name}-{launch_time}"
    assert len(full_job_name) <= JOB_NAME_LENGTH_LIMIT
    return base_job_name, full_job_name


def get_environment(
    entry_point: str,
) -> Dict[str, str]:
    """Returns a dict of environment variables."""
    # Note: We specify `SAGEMAKER_PROGRAM` here so that we can vary
    # the entry point at runtime, versus in Dockerfile.
    environment = {
        "HF_HOME": "/opt/ml/code/cache",
        "HF_TOKEN": get_hf_token_key(),
        "PYTHONPATH": "/opt/ml/code/lbm",
        "SAGEMAKER_PROGRAM": f"/opt/ml/code/lbm/lbm_sagemaker/{entry_point}",
        "WANDB_API_KEY": get_wandb_api_key(),
        "SM_USE_RESERVED_CAPACITY": "1",
        # https://github.shared-services.aws.tri.global/robotics/lbm/issues/981
        "TOKENIZERS_PARALLELISM": "0",
    }
    return environment


def split_cli_overrides(overrides: str) -> dict:
    if not overrides:
        return {}

    out, current_key = {}, None
    for item in overrides.split(","):
        if "=" in item:                       # a normal key=value token
            key, val = item.split("=", 1)     # keep '='s that appear later
            out[key] = val
            current_key = key
        else:                                 # ← NEW: fragment of a list
            if current_key is None:
                raise ValueError(f"Bad override fragment '{item}'")
            out[current_key] += "," + item    # glue “,8]” back onto “[2”

    return out


def overrides_dict_to_str(override_dict: dict) -> str:
    return (
        ",".join([f"{k}={v}" for k, v in override_dict.items()])
        .strip(",")
        .replace(" ", "")
    )


def overrides_dict_to_list(override_dict: str) -> dict:
    return [f"{k}={v}" for k, v in override_dict.items()]


def get_sagemaker_role(*, region):
    """Returns the SageMaker role ARN based on the AWS region."""
    # We append the region name for our `us-west-2` Sagemaker role.
    iam_role_suffix = "" if region == "us-east-1" else "-us-west-2"
    return (
        f"arn:aws:iam::{ML_ACCOUNT_ID}:role/Robotics-LBM-SagemakerAccess"
        f"{iam_role_suffix}"
    )


def get_queue_name(*, region, instance_type, team):
    """Returns the name of the AWS Batch queue based on the AWS region and
    instance type.
    """
    if instance_type not in ALLOWED_INSTANCE_TYPES:
        raise ValueError(f"Invalid instance type: {instance_type}")
    if region not in QUEUE_MAPPER:
        raise ValueError(f"Invalid region: {region}")
    if ALLOWED_INSTANCE_TYPES[instance_type] not in QUEUE_MAPPER[region]:
        raise ValueError(
            f"Invalid instance type for region {region}: {instance_type}"
        )
    if team not in QUEUE_MAPPER[region][ALLOWED_INSTANCE_TYPES[instance_type]]:
        return QUEUE_MAPPER[region][ALLOWED_INSTANCE_TYPES[instance_type]][
            "default"
        ]
    return QUEUE_MAPPER[region][ALLOWED_INSTANCE_TYPES[instance_type]][team]


def check_queue_capacity(region, queue_name, instance_count):
    batch_client = get_batch_boto_client(region=region)
    response = batch_client.describe_job_queues(jobQueues=[queue_name])
    service_environment_arn = response["jobQueues"][0][
        "serviceEnvironmentOrder"
    ][0]["serviceEnvironment"]
    service_environment_name = service_environment_arn.split("/")[-1]
    print(f"Service Environment Name: {service_environment_name}")
    env_response = batch_client.describe_service_environments(
        serviceEnvironments=[service_environment_name]
    )
    queue_capacity = env_response["serviceEnvironments"][0]["capacityLimits"][
        0
    ]["maxCapacity"]
    if queue_capacity < instance_count:
        raise ValueError(
            f"Queue {queue_name} has capacity of {queue_capacity}. "
            f"Requested {instance_count} instances."
        )


def queue_sagemaker_job(
    region,
    instance_type,
    estimator,
    full_job_name,
    priority,
    instance_count=1,
    max_duration_seconds=7 * 24 * 60 * 60,
    team="default",
    inputs=(None,),
):
    assert_training_job_runtime(max_duration_seconds)
    queue_name = get_queue_name(
        region=region, instance_type=instance_type, team=team
    )
    check_queue_capacity(region, queue_name, instance_count)
    boto3.setup_default_session(region_name=region)
    queue = Queue(queue_name=queue_name)
    # We want to show the variable name for clarify but avoid flake8 warning.
    queued_jobs = queue.map(  # noqa: F841
        estimator,
        inputs=inputs,
        job_names=[full_job_name],
        priority=priority,
        share_identifier="default",
        timeout={"attemptDurationSeconds": max_duration_seconds},
    )
    print(
        f"Queued jobs: {queued_jobs} \n "
        f"Job Name: {full_job_name} in Queue : {queue_name} "
    )
    get_job_status(region, queued_jobs, queue_name)


def get_job_status(region, queued_jobs, queue_name):
    batch_client = get_batch_boto_client(region=region)
    for job in queued_jobs:
        job_name = job.job_name
        job_id = job.job_arn.split("/")[-1]
        response = batch_client.describe_service_job(jobId=job_id)
        status = response["status"]
        while status != "RUNNING":
            if status == "FAILED":
                statusReason = ""
                if "statusReason" in response:
                    statusReason = response["statusReason"]
                else:
                    if "attempts" in response:
                        for attempts in response["attempts"]:
                            if "statusReason" in attempts:
                                statusReason = attempts["statusReason"]
                print(
                    f"Job {job_name} is failed for the following reason: "
                    f"{statusReason}"
                )
                break
            if status == "RUNNING":
                print(f"The {job_name} status is {status}.")
                print("Please tail the job logs using sagey")
                break
            print(f"Job {job_name} status is {status}")
            print("Waiting for the resources to be available...")
            time.sleep(10)
            response = batch_client.describe_service_job(jobId=job_id)
            status = response["status"]


def get_checkpoint_s3_uri(
    s3_output_path: str,
    full_job_name: str,
    overrides: Dict,
) -> str:
    """
    Get a checkpoint S3 URI by concatenating `s3_output_path` and
    `full_job_name`. If either `training.resume_checkpoint_path` or
    `+training.resume_checkpoint_path` exists in `overrides.keys()`,
    the checkpoint S3 URI will be replaced with the corresponding value.
    """
    replace = None
    checkpoint_s3_uri = f"{s3_output_path}/{full_job_name}"
    if "training.resume_checkpoint_path" in overrides:
        replace = overrides["training.resume_checkpoint_path"]
    elif "+training.resume_checkpoint_path" in overrides:
        replace = overrides["+training.resume_checkpoint_path"]
    if replace:
        # (kevin-mcgee): if a checkpoint path is provided, we replace the
        # checkpoint_s3_uri with the provided path's root directory. This
        # is to ensure that the checkpoint is saved in the same directory
        # as the rest of the run's output
        checkpoint_s3_uri = replace.rsplit("/", 2)[0]
    return checkpoint_s3_uri
