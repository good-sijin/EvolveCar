FROM carlasim/carla:0.10.0
USER root

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

RUN apt-get update \
    && apt-get install -y \
    libxext6 \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    python3-pip \ 
    zsh \
    sudo \
    git

# Grant carla user with sudo privileges
RUN usermod -aG sudo carla

# New added for disable sudo password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN pip3 install uv

COPY nvidia_icd.json /etc/vulkan/icd.d

USER carla
