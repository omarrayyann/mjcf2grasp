FROM ghcr.io/allenai/cuda:12.8-ubuntu22.04-torch2.6.0

# Install system dependencies including Eigen
RUN apt-get update -q && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends \
        build-essential wget curl git cmake unzip ca-certificates \
        python3-dev \
        libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
        libegl1-mesa libegl1-mesa-dev mesa-utils \
        && \
    apt-get clean && apt-get autoclean -y && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create Conda environment
RUN /opt/miniconda3/bin/conda create -n mjgrasp -y python=3.10

# Add conda environment to PATH
ENV PATH="/opt/miniconda3/envs/mjgrasp/bin:$PATH"

# Copy project files
#COPY . /root/thor-grasp
#WORKDIR /root/thor-grasp

# Install Python dependencies
#RUN pip install --upgrade pip && \
#    pip install -r requirements.txt

## Build Manifold
#RUN if [ -d "external_src/Manifold" ]; then \
#        rm -rf external_src/Manifold/build && \
#        mkdir -p external_src/Manifold/build && \
#        cd external_src/Manifold/build && \
#        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-I/root/thor-grasp/external_src/Manifold/3rd/Eigen" && \
#        make -j$(nproc); \
#    fi

# Use bash login shell by default
ENTRYPOINT [ "bash", "-l" ]