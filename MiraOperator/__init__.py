import sys, os

import torch
import torch.utils.cpp_extension
# Set some default environment provided at setup
# try:
#     # noinspection PyUnresolvedReferences
#     from .envs import persistent_envs
#     for key, value in persistent_envs.items():
#         if key not in os.environ:
#             os.environ[key] = value
# except ImportError:
#     pass

import mira_operator_cpp
print(sys.modules["mira_operator_cpp"].__file__)
mira_operator_cpp.init(os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    torch.utils.cpp_extension.CUDA_HOME)

# Configs
from mira_operator_cpp import (
    set_num_sms,
    get_num_sms
)

# Kernels
from mira_operator_cpp import (
    fp32_add
)