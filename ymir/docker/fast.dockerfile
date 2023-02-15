FROM youdaoyzbx/ymir-executor:ymir1.1.0-yolov7-cu111-tmi

COPY . /app
# pip install required packages
RUN mkdir -p /img-man && \
    mv /app/ymir/img-man/*.yaml /img-man && \
    pip install "git+https://github.com/modelai/ymir-executor-sdk.git@ymir2.1.0" && \
    echo "cd /app && python3 ymir/start.py" > /usr/bin/start.sh

ENTRYPOINT []
CMD bash /usr/bin/start.sh
