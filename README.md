## README

feature：

+ use mujoco physical simulation engine
+ rl algorithm: ppo and policy gradient
+ use burn deeplearning framework that support libtorch, wgpu.
+ support openai gym humanoid_v4 and inverted_pendulum_v4 env
+ support tensorboard
+ Multi-thread and batch sample from env
+ Use bindgen-cli to convert c header to rust

### quick

​	By default, ndarray is used as the backend for burn, and no additional installation is required. 

​	If libtorch is needed as the backend, it needs to be installed. It is recommended to install pytorch.

```````python
pip install torch==2.2.0
```````

​	burn libtorch doc：https://github.com/tracel-ai/burn/blob/main/crates/burn-tch/README.md

##### Ubuntu:
```shell
sudo apt update && sudo apt install gcc build-essential pkg-config cmake clang libxi-dev libxcursor-dev libxinerama-dev libavutil-dev libavformat-dev libavfilter-dev libavdevice-dev libxrandr-dev
# use libtorch:
export LD_LIBRARY_PATH="$PYTORCH_PATH:$LD_LIBRARY_PATH"
# pip show torch
# e.g. /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torch/lib
```

#### macos: 

``````shell
brew install ffmpeg # for video
# use libtorch:
export DYLD_LIBRARY_PATH="$PYTORCH_PATH:$DYLD_LIBRARY_PATH"
# pip show torch
# e.g. ~/miniconda3/lib/python3.12/site-packages/torch/lib
``````

#### performance

在7500f+7900xt(rocm)使用ppo训练humanoid差不多需要5个小时，可以获得差不多8000 reward。

burn backend performance:

Libtorch-cuda > wgpu = macos mps > libtorch cpu > ndarray


#### docker
Amd-gpu rocm docker: https://github.com/pleasewhy/rocm-pytorch-opengl-cpu



