FROM tensorflow/tensorflow:latest-devel-py3
COPY src/ /usr/src
WORKDIR /usr/src
RUN pip install -r requirements.txt
