FROM ghcr.io/allenai/cuda:12.8-ubuntu22.04-torch2.6.0

RUN :\
    && apt-get update -q \
    && export DEBIAN_FRONTEND=nointeractive \
    && apt-get install -y --no-install-recommends \
        build-essential wget curl git cmake unzip ca-certificates \
        python3-dev \
        libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
        libegl1-mesa libegl1-mesa-dev mesa-utils \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -r /var/lib/apt/lists/* \
    && :

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

COPY . /root/mujoco-thor

WORKDIR /root/mujoco-thor

ENV PATH="/root/.local/bin/:$PATH"

ENV PATH="/root/mujoco-thor/.venv/bin:$PATH"

ENV MUJOCO_GL=egl
ENV LIBGL_ALWAYS_SOFTWARE=true

RUN :\
    && uv venv --python 3.10 \
    && curl -o bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl https://download.blender.org/pypi/bpy/bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl \
    && uv pip install bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl \
    # && uv pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/ \
    && uv pip install -e . \
    && uv pip install external_src/dm_control \
    && :

ENTRYPOINT [ "bash", "-l" ]