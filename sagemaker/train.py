"""The training entry point for SageMaker which is functionally identical to
`//lbm/diffusion_policy/scripts/train.py`. The primary difference is to combine
argparse with hydra because a SageMaker launcher, e.g.,
`//lbm/lbm_sagemaker/launcher.py` must only specify a Python entry point with
`hyperparameters` containing all the arguments.
"""

import argparse
import re
import sys

import boto3
import hydra

from diffusion_policy.aws.s3_constants import LBM_S3_BUCKETS, S3_PREFIX
from diffusion_policy.workspace.base_workspace import init_and_run_workspace

# Use line-buffering for both stdout and stderr.
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def assert_s3_access(region):
    bucket = LBM_S3_BUCKETS[region]
    # Remove s3:// prefix and a trailing slash.
    bucket = bucket[len(S3_PREFIX) : -1]
    s3_client = boto3.client("s3")
    assert s3_client.get_object(
        Bucket=bucket, Key=".this_is_s3"
    ), f"Failed accessing {bucket} in {region}"


def main():
    # The frequently-used arguments to override the default training
    # configuration. Note the main function is not wrapped with `@hydra.main`
    # decorator.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--wandb-run-name", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--overrides", type=str, default="")
    args = parser.parse_args()

    if args.overrides:
        # This Regex pattern splits the overrides string by commas that are
        # followed by a key=value pair.
        # For example, "a=1,b=[128,128],c=3" will be split
        # into ["a=1", "b=[128,128]", "c=3"].
        # Additionally, it supports strings like
        # "+a=1,b=[128,128],c=3" by allowing an optional "+" or "~" prefix.
        override_split_pattern = re.compile(
            r",(?=\s*(?:\+\+|\+=|\+|~)?\w+[\w.]*\s*=)"
        )
        extra_overrides = override_split_pattern.split(args.overrides)
        extra_overrides = [eo.strip() for eo in extra_overrides]
    else:
        extra_overrides = []

    overrides = [
        f"training.num_epochs={args.num_epochs}",
        # We log everything (except for checkpoints) into a designated
        # SageMaker direcotry so that they will be synced to S3 at the end.
        "training.output_dir=/opt/ml/output/data",
        f"aws.region={args.region}",
        f"task.dataset.batch_size={args.batch_size}",
        f"task.val_dataset.batch_size={args.batch_size}",
        "logging.mode=online",
        f"logging.name={args.wandb_run_name}",
    ] + extra_overrides

    assert_s3_access(args.region)
    hydra.initialize(config_path="../diffusion_policy/config")
    cfg = hydra.compose(args.config_file, overrides=overrides)

    print("Start SageMaker training!")
    init_and_run_workspace(cfg)


if __name__ == "__main__":
    main()
