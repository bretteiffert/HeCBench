#!/bin/bash

set -exu

bindgen bfs.h -- -xc++ -std=c++14 -Xcompiler -Wall -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda/include -I../../bfs-sycl > bfs_bindings.rs
