# Setup 

## Build docker image and run
```bash
cd docker
docker build -t carlasim/carla:0.9.15-dev .
docker run -it --name dev \
    -privileged --gpus all \
    --net=host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    carlasim/carla:0.9.15-dev /bin/bash

```
## Init project
This project use `uv` to manage the python package. So before you begin to run the project,
you have to install the tool first, with
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Please check [uv homepage](https://docs.astral.sh/uv/) for more information about the tool.

After that, init the project. 
```bash
cd EvolveCar
uv sync
uv pip install -e .
```
It will install all the dependancy including the python itself.

## Run example code 
```bash
uv run evolve_car/simulator/core/carla_env.py
```