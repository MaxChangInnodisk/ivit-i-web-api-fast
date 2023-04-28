# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from .intel_sample import (
    init_intel_samples
)

def init_samples(framework: str):
    """ Initialize Samples into Database """

    if framework == 'openvino':
        init_intel_samples()
    else:
        raise RuntimeError('Unexpected framework: {}'.format(framework))
