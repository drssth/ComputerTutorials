FROM nvcr.io/nvidia/l4t-base:r32.3.1

RUN apt-get update && apt-get install -y apt-utils curl gnupg2
ENV DEBIAN_FRONTEND noninteractive

#
# install dev dependencies
# https://codepyre.com/2019/08/building-tensorflow-object-detection-samples/
#
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-dev \
    libopenmpi-dev \
    git cmake gcc libgfortran4 \
    libhdf5-dev \
    libhdf5-serial-dev \
    libopenblas-base \
    zip unzip \
    hdf5-tools \
    zlib1g-dev libjpeg8-dev \
    python3-dev \
    python3-pip \
    python3-h5py \
    python3-setuptools \
    python3-pil python3-smbus python3-matplotlib

RUN pip3 install Cython
RUN pip3 install tqdm
RUN pip3 install numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta

#
# pytorch
#
RUN wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install torch-1.4.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install -U torchvision

#
# TF-1.15
#
#RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu==1.15.0+nv20.1
RUN wget https://developer.download.nvidia.com/compute/redist/jp/v43/tensorflow-gpu/tensorflow_gpu-1.15.0+nv20.1-cp36-cp36m-linux_aarch64.whl
RUN pip3 install tensorflow_gpu-1.15.0+nv20.1-cp36-cp36m-linux_aarch64.whl

ENTRYPOINT /bin/bash