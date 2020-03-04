# Build a docker image for Jetson system

This article shows the steps to build a docker image for Nvidia jetson system.

Requirements:
- Nvidia Jetson board, I am using a [Jetson Nano](https://developer.nvidia.com/buy-jetson?product=jetson_nano&location=US) board
- [Jetpack](https://developer.nvidia.com/embedded/jetpack) installed on the jetson system. I recommend install Jetpack v4.2 or v4.3. This article is tested on Jetpack v4.2. I assume v4.3 should be similar. (Please follow the official guild to install Jetpack. Otherwise, some steps may not apply.)

Now we boot from Jetson. Jetpack comes with docker, nvidia runtime installed. To verify docker installation:
```
$ docker --version
Docker version 18.09.7, build 2d0083d
```

To verify docker nvidia runtime

```$ nvidia-container-runtime --version
runc version spec: 1.0.1-dev
```

If nvidia runtime is not available, we could install docker runtime with

```
sudo apt install nvidia-container-runtime
```
Note that docker mush be run with root permission.
We could use the following command to enable root permission for the current user:
```
sudo usermod -aG docker $USER
```
Note that this command needs a reboot/re-login to take effect.

Now that the requists are all met, we need to set nvidia runtime as default.
Docker daemon setting is located at ```/etc/docker/daemon.json```

I edit this file by using 
```
$ sudo apt install nano
$ sudo nano /etc/docker/daemon.json
```

By default the setting should be

```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```
This means we could use ```docker run --runtime=nvidia ...``` to enable nvidia-container-runtime.

To make ```nvidia``` default, we need to change it to

```
{
    "default-runtime":"nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

To verify nvidia is properly enabled, we could run the following commands to give it a test.

```
$ docker run --name l4t-base -it nvcr.io/nvidia/l4t-base:r32.3.1 bash
root@1a3ca6007e7c:/# find / -iname libnvToolsExt.so*
/usr/local/cuda-10.0/targets/aarch64-linux/lib/libnvToolsExt.so.1
/usr/local/cuda-10.0/targets/aarch64-linux/lib/libnvToolsExt.so.1.0.0
/usr/local/cuda-10.0/targets/aarch64-linux/lib/libnvToolsExt.so
/usr/local/cuda-10.0/doc/man/man7/libnvToolsExt.so.7
root@1a3ca6007e7c:/#
```

Note that the above commands first run the official l4t base container interactively. Official tag names could be find on [Nvidia NGC](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-base).
Next, it finds local cuda library ```libnvToolsExt.so```, which should be installed on the host machine. If there is no output with ```find / -iname libnvToolsExt.so*```, you need to roll back and double check docker and nvidia runtime. If docker container name "l4t-base" is occupied, for example, you see an error message such as ```docker: Error response from daemon: Conflict. The container name "/l4t-base" is already in use by container ...```; then, you can run ```docker rm l4t-base``` to remove the container or replace it with a new name.

Next, we write the ```Dockerfile```:
```
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
```

I came up with the above code with two references. One is [Official Tensorflow for Jetson Nano](https://devtalk.nvidia.com/default/topic/1048776/jetson-nano/official-tensorflow-for-jetson-nano-/); the other is [PyTorch for Jetson Nano](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano-version-1-4-0-now-available/). These two installation methods could be universally applied on other Jetson systems. However, there are several modifications.
1. First, ```apt update``` is required to install other packages. Otherwise the error comes as no installation candidate.
2. Second, ```curl gnupg2``` is there to silence time zone setting interactives. I put ```apt-utils``` there also to fix message ```debconf: delaying package configuration, since apt-utils is not installed```; but it may not be effective.
3. [Official Tensorflow for Jetson Nano](https://devtalk.nvidia.com/default/topic/1048776/jetson-nano/official-tensorflow-for-jetson-nano-/) installs ```h5py``` with ```pip3```, but it cannot find the HDF5 binary when building the docker image. It has to be installed with ```apt-get install python3-h5py```.
4. ```libboost-dev``` and ```libopenmpi-dev``` are not pre-installed in the official docker container and have to be installed in prior to tensorflow and pytorch.
5. I used to put ```apt-get install libopenblas-base``` after tensorflow installation, but maybe due to the long time run of the tensorflow installation, it could not find installation candidate and aborted.
6. Tensorflow cannot be installed with the official method ```pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 tensorflow-gpu==1.15.0+nv20.1```. You have to download it first and then install with ```pip3```.

Once you place this ```Dockerfile``` in a director, I put it under a folder named ```pytorch```:
```
.../pytorch$ ls
Dockerfile
```
Then, I build it with command
```
docker build -t l4t-tf-torch .
```

Then, after finished, you should see ```Successfully tagged l4t-tf-torch:latest```.

You have successfullly build this docker image on that includes tensorflow and pytorch.
To verify installation, you could run it with ```docker run -it l4t-tf-torch bash``` and ```python3 -c "import torch;import tensorflow as tf```.

You can also find the container ID by ```docker ps -a```, commit the updates with ```docker commit <container ID> <your docker account>/<your docker repo>:latest```, and push the image to docker hub with ```docker push <your docker account>/<your docker repo>:latest```.
