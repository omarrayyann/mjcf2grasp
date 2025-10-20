# FROM gcr.io/ai2-beaker-core/public/d1tadgm04onnhjrit6m0 AS conda_env_builder
# FROM gcr.io/ai2-beaker-core/public/d3emt23b43ac73fgb4ig AS conda_env_builder
# # corresponds to beaker image pull ai2/cuda12.8-ubuntu22.04-torch2.6.0 as of 19AUG2025 (why rebuild them so frequently??)

FROM gcr.io/ai2-beaker-core/public/d3a6kl3b43ac738sho4g AS conda_env_builder
# corresponds to beaker image pull ai2/cuda12.8-dev-ubuntu22.04-torch2.7.1 as of 08OCT2025

ENV APP_HOME /root/mujoco-thor
WORKDIR $APP_HOME

# Update package lists and install git-lfs and build tools
RUN apt-get update && \
    apt-get install -y git-lfs ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN /opt/miniconda3/bin/conda create -n mjthor -y python=3.10

RUN /opt/miniconda3/bin/conda install -n mjthor -y -c conda-forge setuptools wheel ninja
RUN /opt/miniconda3/bin/conda clean -ya

FROM conda_env_builder AS requirements_installer

ENV PYTHON=/opt/miniconda3/envs/mjthor/bin/python
ENV PIP=/opt/miniconda3/envs/mjthor/bin/pip

# CUDA development tools are already installed in base image
# Verify they're available
RUN echo "=== Verifying CUDA from base image ===" && \
    nvcc --version

# Set CUDA architecture for A6000, L40, A100, and H100
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CUDA_VISIBLE_DEVICES=0

# MuJoCo headless rendering environment variables
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV MUJOCO_EGL_DEVICE_ID=0

# Copy only pyproject.toml first for better layer caching
COPY ./pyproject.toml $APP_HOME/pyproject.toml

# Install dependencies and clean up in one layer to reduce size
RUN ( \
    export PIP_SRC=/opt/miniconda3/envs/mjthor/pipsrc; \
    cd $APP_HOME \
    && $PIP install --no-cache-dir --no-deps -e . \
    && $PIP install --no-cache-dir --upgrade-strategy only-if-needed \
        "torch>=2.7.0,<2.8.0" "torchvision>=0.22.0,<0.23.0" \
        "numpy<2.0.0" "pillow==9.3.0" \
        --index-url https://download.pytorch.org/whl/cu128 \
    && $PIP install --no-cache-dir --group docker -e $APP_HOME \
    && $PIP install --no-cache-dir --upgrade "typing-extensions>=4.14.1" \
    && $PIP cache purge \
)

# Verify PyTorch installation in conda env
RUN echo "=== Verifying PyTorch in conda environment ===" && \
    $PYTHON -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

FROM requirements_installer AS curobo_installer

# Build args for cuRobo installation
ARG GITHUB_TOKEN
# Pinned to a specific commit for reproducibility - update as needed
ARG CUROBO_COMMIT=417c995647fcb173a2bc094d1284b2a4f4b000ad

# Clone and install cuRobo from AllenAI fork (private repo)
WORKDIR /workspace
RUN git clone https://${GITHUB_TOKEN}@github.com/allenai/curobo.git

WORKDIR /workspace/curobo
RUN git checkout ${CUROBO_COMMIT}

# Set environment variables for cuRobo compilation with multi-GPU support
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV CUDA_LAUNCH_BLOCKING=1

# Install cuRobo and clean up in one layer
RUN $PIP install --no-cache-dir . --no-build-isolation \
    && $PIP cache purge \
    && rm -rf /workspace/curobo/.git \
    && find /workspace/curobo -type f -name "*.pyc" -delete \
    && find /workspace/curobo -type d -name "__pycache__" -delete

# Test basic import and functionality (CPU-only, doesn't require GPU drivers)
RUN CUDA_VISIBLE_DEVICES="" $PYTHON -c \
    "print('=== Testing PyTorch ==='); \
    import torch; \
    print('PyTorch version:', torch.__version__); \
    print('CUDA available:', torch.cuda.is_available()); \
    print('CUDA version:', torch.version.cuda); \
    x = torch.randn(3, 3); \
    y = torch.matmul(x, x); \
    print('PyTorch tensor operations working:', y.shape); \
    print('=== Testing cuRobo ==='); \
    import curobo; \
    print('cuRobo imported successfully!'); \
    from curobo.types.math import Pose; \
    print('cuRobo core types imported successfully!'); \
    print('=== Installation Complete ==='); \
    print('cuRobo installation verified - ready for GPU usage!')" \


FROM curobo_installer as resource_manager

ARG GITHUB_TOKEN

# Install mujoco-thor-resources and clean up in one layer
RUN --mount=type=secret,id=github_token,env=GITHUB_TOKEN ( \
    export PIP_SRC=/opt/miniconda3/envs/mjthor/pip_src; \
    $PIP install --no-build-isolation --no-cache-dir -e git+https://${GITHUB_TOKEN}@github.com/allenai/mujoco-thor-resources.git#egg=mujoco_thor_resources \
    && $PIP cache purge \
)


FROM curobo_installer AS final

# Set working directory back to app home
WORKDIR $APP_HOME

# Note: Source code is NOT copied into image - mount it at runtime at $APP_HOME
# This keeps the image smaller and more reusable

# Set default MuJoCo-THOR assets directory (can be overridden at runtime)
ENV WEKA_DEFAULT_MJCTHOR_ASSETS_DIR=/weka/prior/datasets/robomolmo/mjthor_resources/

# Set PYTHONPATH to include mounted code directory
ENV PYTHONPATH=$APP_HOME:$PYTHONPATH

COPY --from=resource_manager /opt/miniconda3/envs/mjthor /opt/miniconda3/envs/mjthor

# Aggressive cleanup to reduce image size
RUN /opt/miniconda3/bin/conda clean -ya \
    && $PIP cache purge \
    && find /opt/miniconda3 -type f -name "*.pyc" -delete \
    && find /opt/miniconda3 -type d -name "__pycache__" -delete \
    && find /opt/miniconda3 -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/miniconda3 -type d -name "test" -exec rm -rf {} + 2>/dev/null || true \
    && rm -rf /opt/miniconda3/pkgs/* \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* /var/tmp/* \
    && touch /root/.git-credentials

# The -l flag makes bash act as a login shell and load /etc/profile, etc.
ENTRYPOINT ["bash", "-l"]
