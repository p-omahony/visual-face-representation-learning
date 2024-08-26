FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install flask==3.0.3 gunicorn==22.0.0 boto3

ARG WANDB_KEY
ENV WANDB_KEY=$WANDB_KEY
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY face_learner /opt/program

WORKDIR /opt/program
