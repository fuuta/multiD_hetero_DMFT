# Overview

Python code for paper:

Tomita F, Teramae J-N. Dynamical mean-field theory for a highly heterogeneous neural population [Internet]. arXiv [nlin.CD]. Available from: http://arxiv.org/abs/2412.10062

# Requirement
require any GPUs to fast numerical simulation

# Prepare environment
we use podman and uv to build clean environment.

1. install podman
2. change `SOURCE_PROJECT_DIR` in `.devcontainer/.env`
3. build and run container using VSCODE devcontainer extention or podman-compose

# Directory structure

multid_rnn: main code for simulation and visualization

results: destination to save simulation results

scripts: script to reproduce figures in paper (some parts are in preparation for release)

TEMP: temporary information e.g. logs
