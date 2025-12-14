# --- 기존 10.2/7/18.04 -> 11.1/8/20.04로 정렬 (torch cu111와 일치) ---
ARG CUDA=11.1.1
ARG CUDNN=8
ARG UBUNTU=20.04

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON=3.8.7
ENV PYTHON_ROOT=/root/local/python-$PYTHON
ENV PATH=$PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT=/root/.pyenv
ENV POETRY=1.2.1

# --- 여기서 apt-key 줄 제거 + apt-get로 통일 ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git less \
      libbz2-dev libffi-dev libgl1 liblzma-dev libncurses5-dev libncursesw5-dev \
      libreadline-dev libsqlite3-dev libssl-dev llvm make openssh-client \
      tk-dev tmux unzip vim wget xz-utils zip zlib1g-dev \
      gnupg2 dirmngr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# python build
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    $PYENV_ROOT/plugins/python-build/install.sh && \
    /usr/local/bin/python-build -v $PYTHON $PYTHON_ROOT && \
    rm -rf $PYENV_ROOT

# ✅ pip가 manylinux_2_28 wheel 인식하도록 업그레이드
RUN $PYTHON_ROOT/bin/python -m ensurepip && \
    $PYTHON_ROOT/bin/python -m pip install -U pip setuptools wheel

# python3/pip 명령이 항상 네 PYTHON_ROOT를 가리키게 강제
RUN ln -sf $PYTHON_ROOT/bin/python /usr/local/bin/python3 && \
    ln -sf $PYTHON_ROOT/bin/pip /usr/local/bin/pip

ENV HOME=/root
WORKDIR $HOME

# install poetry (python3가 위에서 보장됨)
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY python3 -
ENV PATH=$HOME/.local/bin:$PATH

WORKDIR /root/workspace
COPY pyproject.toml poetry.lock poetry.toml ./

RUN mkdir -m 700 $HOME/.ssh && ssh-keyscan github.com > $HOME/.ssh/known_hosts
RUN --mount=type=ssh poetry install --no-root

# torch는 python -m pip로 (PATH 혼선 방지)
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
      -f https://download.pytorch.org/whl/torch_stable.html
RUN python -m pip install --no-cache-dir fastapi uvicorn python-multipart
