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
# init_ivit_env()

# OpenVINO
def vino_init(type:str, params:dict):
    """ Initialize OpenVINO """
    if type == 'CLS':
        from ivit_i.core.models import iClassification
        model = iClassification(
            model_path = params['model_path'],
            label_path = params['label_path'],
            device = params['device'],
            confidence_threshold = params['confidence_threshold'],
            topk = params['topk'],
        )
    elif type == 'OBJ':
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

def get_ivit_api(framework:Literal['openvino', 'nvidia', 'jetson', 'xilinx', 'hailo']):
    
    map = {
        "openvino": vino_init
    }

    return map[framework]

