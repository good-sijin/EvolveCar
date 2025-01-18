FROM carlasim/carla:0.9.15
USER root

RUN apt-get update \
    && apt-get install -y \
    zsh \
    sudo \
    git \
    wget \
    curl \
    autojump

# Grant carla user with sudo privileges
RUN usermod -aG sudo carla

# New added for disable sudo password
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Leave it alone, just do it muanully.
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh

USER carla

