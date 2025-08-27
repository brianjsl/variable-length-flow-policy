"""The launcher to submit a SageMaker job. It serves as the highest-level
configuration, such as the number of nodes and the AWS settings.
"""

import argparse
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from eval_automation.interface import eval_automation_add_run

from sagemaker.util import (
    ALLOWED_INSTANCE_TYPES,
    assert_sagemaker_profile,
    assert_us_west_2_region,
    get_checkpoint_s3_uri,
    get_environment,
    get_job_names,
    get_kms_encryption_key,
    get_s3_output_path,
    get_sagemaker_role,
    get_sagemaker_tags,
    maybe_set_wandb_group_name,
    overrides_dict_to_str,
    queue_sagemaker_job,
    split_cli_overrides,
)

DEFAULT_TRAINING_CONFIG_DIR = "experiment_configs/aloha"
DEFAULT_TRAINING_CONFIG_NAME = "transformer_aloha"


def main():
    assert_sagemaker_profile()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=str,
        default=DEFAULT_TRAINING_CONFIG,
        help=f"The default is {DEFAULT_TRAINING_CONFIG}.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_TRAINING_CONFIG_NAME,
        help=f"The default is {DEFAULT_TRAINING_CONFIG_NAME}.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=500,
        help="The default is 500.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=7,
        help="The maximums number of training days.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=36,
        help="The default is 36.",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="The default is 1.",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="p4de",
        choices=["p5", "p4de"],
        help="The default is 'p4de'.",
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        default="variable-length-flow-policy",
        help="The default is 'variable-length-flow-policy'.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-west-2",
        choices=["us-east-1", "us-west-2"],
        help="The default is `us-west-2`.",
    )
    parser.add_argument(
        "--config-overrides",
        type=str,
        default="",
        help="Overrides to be passed to the hydra config. e.g. "
        "`dataset.storage_dirs_list=[...],action.shape=[...]`.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="The priority of the job in a scale from 0 (lower) to 9999 "
        "(higher). Default to 5.",
    )
    parser.add_argument(
        "--team",
        type=str,
        default="default",
        choices=["default", "mvdp", "vla", "das", "ml", "bd1", "bd2", "bd", "testing"],
        help="The team to which the job belongs. "
    )

    args = parser.parse_args()

    assert_us_west_2_region(region=args.region)

    # Some variables a PyTorch Estimator would need.
    sagemaker_session = sagemaker.Session(
        boto3.Session(region_name=args.region)
    )
    entry_point = Path(__file__).parent / "train.py"
    s3_output_path = get_s3_output_path(
        region=args.region,
        is_main_branch=args.image_uri == "variable-length-flow-policy",
    )
    base_job_name, full_job_name = get_job_names(
        is_main_branch=args.image_uri == "variable-length-flow-policy", task_name=None
    )
    # We only expose these args that are frenquently changed between runs.
    hyperparameters = {
        "config-file": args.config_file,
        # Change this to a smaller number for instances with limited GPU
        # memory. Note: For p5 and p4de instances (with 80G GPU memory), and
        # p4d instances (40G), the optimal `batch-size`s are 36 and 18,
        # respectively.
        "batch-size": args.batch_size,
        # Change this for quick testing. Currently, it's set to a large enough
        # number to run for at least 7 days (maximum runtime).
        "num-epochs": args.num_epochs,
        # Note: A temporary solution to alter AWS region for S3 data.
        "region": args.region,
        # We overwrite the default W&B run name so that it's better associated
        # with the S3 directory and the SageMaker run.
        "wandb-run-name": full_job_name,
    }
    # TODO(krishnan): add support for building hydra overrides from config
    overrides = split_cli_overrides(args.config_overrides)
    overrides = maybe_set_wandb_group_name(overrides, full_job_name)
    override_str = overrides_dict_to_str(overrides)
    if override_str:
        hyperparameters["overrides"] = override_str
    environment = get_environment(entry_point="train.py")
    checkpoint_s3_uri = get_checkpoint_s3_uri(
        s3_output_path, full_job_name, overrides
    )

    estimator = PyTorch(
        base_job_name=base_job_name,
        # The entry point that launches the training script with options.
        entry_point=str(entry_point),
        hyperparameters=hyperparameters,
        role=get_sagemaker_role(region=args.region),
        # Point to the image to use in ECR.
        image_uri=(
            f"124224456861.dkr.ecr.{args.region}.amazonaws.com/{args.image_uri}"  # noqa
        ),
        instance_count=args.instance_count,
        instance_type=ALLOWED_INSTANCE_TYPES[args.instance_type],
        environment=environment,
        sagemaker_session=sagemaker_session,
        # Set the maximum training time (seconds).
        max_run=args.max_days * 24 * 60 * 60,
        output_kms_key=get_kms_encryption_key(region=args.region),
        # The S3 prefix URI where custom code is uploaded.
        code_location=s3_output_path,
        # The S3 location for saving the training result (model artifacts and
        # output files).
        output_path=s3_output_path,
        # These two arguments are needed for checkpoint saving during the run.
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path="/opt/ml/checkpoints",
        keep_alive_period_in_seconds=300,
        tags=get_sagemaker_tags(),
        distribution={"torch_distributed": {"enabled": True}},
    )

    # Use the batch queueing feature, i.e., submit the job to a queue and
    # it will be executed based on the capacity and priority.
    queue_sagemaker_job(
        args.region,
        args.instance_type,
        estimator,
        full_job_name,
        args.priority,
        args.instance_count,
        args.max_days * 24 * 60 * 60,
        args.team,
    )


if __name__ == "__main__":
    main()
