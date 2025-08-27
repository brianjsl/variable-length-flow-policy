# NOTE: REGION_PLACEHOLDER will be replaced by your outer script.
ARG BASE_IMAGE=763104351884.dkr.ecr.REGION_PLACEHOLDER.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker
FROM ${BASE_IMAGE}

# Use bash with fail-fast
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Build args for flexibility
ARG REQUIREMENTS_FILE=requirements.txt
ARG CONDA_ENV_NAME=robodiff-lh

# --- OS deps + Mambaforge (mamba/conda) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates bzip2 bindfs nfs-common vim && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
      -o /tmp/mambaforge.sh && \
    bash /tmp/mambaforge.sh -b -p /opt/conda && \
    rm -f /tmp/mambaforge.sh && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda clean -afy

# Copy minimal build inputs early for layer caching
COPY ./${REQUIREMENTS_FILE} /opt/ml/code/${REQUIREMENTS_FILE}
COPY ./conda_environment.yaml /opt/ml/code/conda_environment.yaml

# --- Create the conda env with mamba ---
RUN /opt/conda/bin/mamba env create -f /opt/ml/code/conda_environment.yaml && \
    echo "export PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:\$PATH" >/etc/profile.d/robodiff-env.sh

# Make the env “active” for all subsequent layers and at runtime
ENV CONDA_PREFIX=/opt/conda/envs/${CONDA_ENV_NAME}
ENV CONDA_DEFAULT_ENV=${CONDA_ENV_NAME}
ENV PATH=${CONDA_PREFIX}/bin:/opt/conda/bin:${PATH}

# (Optional) Strip torch from requirements to avoid conflicts with base image/conda env pinning.
# If you WANT torch via conda yaml, keep it there; don't also install via pip.
RUN if grep -q '^torch' /opt/ml/code/${REQUIREMENTS_FILE}; then \
      grep -v '^torch' /opt/ml/code/${REQUIREMENTS_FILE} > /opt/ml/code/req_no_torch.txt && \
      mv /opt/ml/code/req_no_torch.txt /opt/ml/code/${REQUIREMENTS_FILE}; \
    fi

# --- Python deps into the conda env ---
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /opt/ml/code/${REQUIREMENTS_FILE} && \
    python -m pip install --no-cache-dir packaging ninja && \
    python -m pip install --no-cache-dir flash-attn --no-build-isolation && \
    pip list -v

# SageMaker setup
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

# Copy full source last for better caching
COPY ./ /opt/ml/code/

# Prevent SageMaker from trying to pip-install at runtime
RUN rm -f /opt/ml/code/requirements*

# Optional: copy AWS creds for local builds
ARG local_build
RUN if [ "${local_build:-}" = "TRUE" ] ; then \
      mkdir -p /root/.aws/ && cp /opt/ml/code/lbm/setup/aws/config /root/.aws/; \
    fi
