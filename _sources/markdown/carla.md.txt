# Carla
We use carla as our simulator. However it is really annoying to setup carla inside docker.
So, we document the procedure about how to make it work.

## Step by step
### Update nvidia driver version to 550. 
Since there is known bug in lower version, just upgrade to latest one.
```bash
sudo apt install nvidia-driver-550
```
### Upgrade nvidia-container-toolkit.
Please refer to the document in: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
However, some network problem make me turn to another way. Download the `.deb` file from github and install every package manually.
```
# Install them in order.
sudo dpkg -i libnvidia-container-dev_1.17.3-1_amd64.deb
sudo dpkg -i libnvidia-container1_1.17.3-1_amd64.deb
sudo dpkg -i libnvidia-container-dev_1.17.3-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit-base_1.17.3-1_amd64.deb
sudo dpkg -i libnvidia-container-tools_1.17.3-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit_1.17.3-1_amd64.deb
```

### Rebuild the docker. 
```dockerfile
FROM carlasim/carla:0.10.0
USER root

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

RUN apt-get update \
    && apt-get install -y \
    libxext6 \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools

# You must copy the one in your local computer into current directory.
# cp /usr/share/vulkan/icd.d/nvidia_icd.json .
COPY nvidia_icd.json /etc/vulkan/icd.d
USER carla
```

### Run
``` bash
docker run --privileged --gpus all \
  --net=host -e DISPLAY=$DISPLAY  \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  carlasim/carla:0.10.0-fixed /bin/bash \
  CarlaUnreal.sh \
  -windowed -ResX=600 -ResY=600 --carla-rpc-port=22912 -quality-level=Low
```