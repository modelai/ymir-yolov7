ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# cuda11.1 + pytorch 1.9.0 + cudnn8 not work!!!
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime
# support YMIR=1.0.0, 1.1.0 or 1.2.0
ARG YMIR="1.1.0"
ENV PYTHONPATH=.
ENV YMIR_VERSION=$YMIR

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

# change apt and pypy mirrors
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install linux package
RUN	apt-get update && apt-get install -y gnupg2 git libglib2.0-0 \
    libgl1-mesa-glx libsm6 libxext6 libxrender-dev curl wget zip vim \
    build-essential ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy file from host to docker and install requirements
COPY . /app

# Download pretrained weight and font file
RUN cd /app && mkdir -p /img-man && mv /app/ymir/img-man/*-template.yaml /img-man/ \
    && mkdir -p /root/.config/Ultralytics \
    && wget https://ultralytics.com/assets/Arial.ttf -O /root/.config/Ultralytics/Arial.ttf \
    && pip install -r /app/requirements.txt \
    && pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir1.3.0" \
    && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt \
    && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt \
    && wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt \
    && echo "python3 /app/start.py" > /usr/bin/start.sh

WORKDIR /app

# overwrite entrypoint to avoid ymir1.1.0 import docker image error.
ENTRYPOINT []
CMD bash /usr/bin/start.sh
