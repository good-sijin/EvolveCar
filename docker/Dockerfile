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
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

COPY nvidia_icd.json /etc/vulkan/icd.d

USER carla
