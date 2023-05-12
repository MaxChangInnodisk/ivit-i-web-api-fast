# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from ..common import init_ivit_env

CLS = "CLS"
OBJ = "OBJ"
SEG = "SEG"

# OpenVINO
def vino_init(type:str, params:dict):
    """ Initialize OpenVINO """
    if type == CLS:
        from ivit_i.core.models import iClassification
        model = iClassification(
            model_path = params['model_path'],
            label_path = params['label_path'],
            device = params['device'],
            confidence_threshold = params['confidence_threshold'],
            topk = params.get('topk', 1),
        )
    elif type == OBJ:
        from ivit_i.core.models import iDetection
        model = iDetection(
            model_path = params["model_path"],
            label_path = params["label_path"],
            device = params["device"],
            architecture_type = params["arch"],
            anchors = params["anchors"],
            confidence_threshold = params["confidence_threshold"]
        )
    else:
        raise RuntimeError('Unexpect Type: {}'.format(type))
    
    return model

# Jetson
def jetson_init():
    pass

# dGPU
def dgpu_init():
    pass

# Hailo
def hailo_init():
    pass

# Xilinx
def xlnx_init(type:str, params:dict):
    """ Initialize Xilinx """
    if type == CLS:
        from ivit_i.core.models import iClassification
        model = iClassification(
            model_path = params['model_path'],
            label_path = params['label_path'],
            # device = params['device'],
            confidence_threshold = params['confidence_threshold'],
            topk = params.get('topk', 1),
        )
    elif type == OBJ:
        from ivit_i.core.models import iDetection
        model = iDetection(
            model_path = params["model_path"],
            label_path = params["label_path"],
            # device = params["device"],
            # architecture_type = params["arch"],
            anchors = params["anchors"],
            confidence_threshold = params["confidence_threshold"]
        )
    else:
        raise RuntimeError('Unexpect Type: {}'.format(type))
    
    return model



def get_ivit_api(framework:Literal['openvino', 'tensorrt', 'jetson', 'vitis-ai', 'hailort']):
    
    map = {
        "openvino": vino_init,
        "tensorrt": jetson_init,
        "vitis-ai": xlnx_init,
        "hailort": hailo_init
    }

    return map[framework]

