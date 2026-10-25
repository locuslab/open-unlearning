FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install Python 3.11 from deadsnakes (Ubuntu 22.04 ships 3.11.0rc1, which has
# inspect bugs that break triton's JIT source parsing).
# Bootstrap pip via get-pip.py so we get upstream pip/setuptools (Ubuntu's
# python3-pip carries Debian patches that break pyproject builds).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
       python3.11 python3.11-dev python3.11-distutils git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install dependencies from requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install lm-eval and hf_transfer
RUN pip install --no-cache-dir lm-eval==0.4.11

# Install flash-attn (pre-built wheel for cu128 + torch2.9, avoids slow compilation)
RUN pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"

WORKDIR /workspace