#!/bin/bash

#Linux Script Installation Ubuntu 22.04 wsl

# Update and upgrade the system
sudo apt update && sudo apt upgrade -y

# Install required dependencies
sudo apt install python3 python3-pip python3-venv build-essential cmake -y

# Set up a virtual environment
python3 -m venv instructlab-env
source instructlab-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Set environment variables for CMake arguments
export CMAKE_ARGS="-DLLAMA_CUDA=on -DLLAMA_NATIVE=off"

# Install instructlab
pip install instructlab

# Install additional dependencies if needed
# pip install -r /path/to/instructlab/requirements.txt

# Verify installation