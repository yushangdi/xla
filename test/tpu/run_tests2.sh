#!/bin/bash
set -xue

pip install expecttest rich
# Torchvision
pip install --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@2c127da8b5e2e8f44b50994c6cb931bcca267cfe"
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
git clone -b lsiyuan/v5p-test https://github.com/pytorch/xla.git 
cd xla
export PJRT_DEVICE=TPU
export TPU_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/torch_xla/lib/libtpu.so
test/tpu/run_tests.sh