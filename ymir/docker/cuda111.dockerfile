ARG baseimage=20.12-py3
# yolov7 recommended for 21.08-py3, here we use py3.8 + cuda11.1.1 + pytorch1.8.0
# view https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12 for details
FROM nvcr.io/nvidia/pytorch:${baseimage}
ENV PYTHONPATH=.

# apt install required packages
RUN apt update && \
    apt install -y zip htop vim libgl1-mesa-glx

COPY . /yolov7
# pip install required packages
RUN pip install -r /yolov7/requirements.txt && \
    mkdir -p /img-man && \
    mv /yolov7/ymir/img-man/*.yaml /img-man && \
    echo "cd /yolov7 && python3 ymir/start.py" > /usr/bin/start.sh

ENTRYPOINT []
CMD bash /usr/bin/start.sh
