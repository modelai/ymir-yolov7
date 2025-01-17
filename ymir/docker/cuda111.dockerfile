ARG baseimage=20.12-py3
# yolov7 recommended for 21.08-py3, here we use py3.8 + cuda11.1.1 + pytorch1.8.0
# view https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12 for details
FROM nvcr.io/nvidia/pytorch:${baseimage}
ENV PYTHONPATH=.
ENV YMIR_VERSION=$YMIR

# change apt and pypy mirrors
RUN sed -i 's#http://archive.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com#https://mirrors.ustc.edu.cn#g' /etc/apt/sources.list \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# apt install required packages
RUN apt update && \
    apt install -y zip htop vim libgl1-mesa-glx

# install ymir-exc
RUN pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0"

COPY . /yolov7
# pip install required packages
RUN pip install -r /yolov7/requirements.txt && \
    mkdir -p /img-man && \
    mv /yolov7/ymir/img-man/*.yaml /img-man && \
    echo "cd /yolov7 && python3 ymir/start.py" > /usr/bin/start.sh

WORKDIR /yolov7
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt && \
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt && \
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt


# overwrite entrypoint to avoid ymir1.1.0 import docker image error.
ENTRYPOINT []
CMD bash /usr/bin/start.sh
