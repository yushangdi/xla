# add `build_util` to import path
import os
import sys
sys.path.append(os.path.join(os.dirname(__file__), '..', '..'))

import build_util
import setuptools

build_util.bazel_build('@xla//xla/pjrt/c:pjrt_c_api_gpu_plugin.so',
                       'torch_xla_cuda_plugin/lib', ['--config=cuda'])

setuptools.setup()
