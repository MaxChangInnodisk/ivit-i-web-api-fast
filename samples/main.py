# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from .intel_sample import init_intel_samples
from .xlnx_sample import init_xlnx_samples

def init_samples(framework: str):
    """ Initialize Samples into Database """

    sample_table = {
        'openvino': init_intel_samples,
        'vitis-ai': init_xlnx_samples
    }
    func = sample_table.get(framework, None) 
    
    if func is None: 
        raise RuntimeError('Unexpected framework: {}'.format(framework))

    func()
