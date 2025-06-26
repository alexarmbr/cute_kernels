FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# setup python3.12
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.12 python3.12-venv python3.12-dev python3-pip
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# setup venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc


RUN pip install \
    cuda-bindings==12.9.0 \
    cuda-python==12.9.0 \
    nvidia-cutlass-dsl==4.0.0 \
    torch==2.7.0