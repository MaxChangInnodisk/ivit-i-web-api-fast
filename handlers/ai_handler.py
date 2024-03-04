# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import logging as log
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from .ivit_handler import iModel

try:
    from ..common import init_ivit_env
    from ..common import SERV_CONF
except:
    from common import init_ivit_env
    from common import SERV_CONF

CLS = "CLS"
OBJ = "OBJ"
SEG = "SEG"

class InvalidModelTypeError(Exception):
    def __init__(self, message) -> None:
        self.message = message


# OpenVINO
def ivit_i_intel(type: Literal["CLS", "OBJ", "SEG"], params:dict) -> iModel:
    """Initialize iVIT-I Intel Function

    Args:
        type (Literal[&quot;CLS&quot;, &quot;OBJ&quot;, &quot;SEG&quot;]): the model type
        params (dict): the model setting

    Raises:
        InvalidModelTypeError: get unexpected model type

    Returns:
        iModel: return model which inherit iModel
    """
    if type == CLS:
        from ivit_i.core.models import iClassification
        model = iClassification(
            model_path = params['model_path'],
            label_path = params['label_path'],
            device = params['device'],
            confidence_threshold = params['confidence_threshold'],
            topk = params.get('topk', 3)
        )
    elif type == OBJ:
        from ivit_i.core.models import iDetection
        
        # Update arch for intel platform
        orig_arch = params["arch"]
        correct_arch = orig_arch
        
        if 'torch' in orig_arch:
            from ivit_i.core.models import iDetectionOnnx
            correct_arch = 'torch-yolo'
            log.warning(f'Detect arch is ({orig_arch}), auto convert to {correct_arch}, only for yolov4-tiny')
            return iDetectionOnnx(
                model_path = params["model_path"],
                label_path = params["label_path"],
                device = params["device"],
                architecture_type = correct_arch,
                anchors = params["anchors"],
                confidence_threshold = float(params["confidence_threshold"])
            )
            
        
        if 'yolov4' in orig_arch:
            correct_arch = 'yolov4'
            log.warning(f'Detect arch is ({orig_arch}), auto convert to {correct_arch}')
            
        elif 'yolov3' in orig_arch:
            correct_arch = 'yolo'
            log.warning(f'Detect arch is ({orig_arch}), auto convert to {correct_arch}')
        
        
        # Support yolo,yolov4,yolof,yolox,yolov3-onnx
        model = iDetection(
            model_path = params["model_path"],
            label_path = params["label_path"],
            device = params["device"],
            architecture_type = correct_arch,
            anchors = params["anchors"],
            confidence_threshold = float(params["confidence_threshold"])
        )
    else:
        raise InvalidModelTypeError('Unexpect Type: {}'.format(type))
    
    return model

# Jetson
def ivit_i_jetson():
    pass

# dGPU
def ivit_i_dgpu(type: Literal["CLS", "OBJ", "SEG"], params:dict) -> iModel:
    """Initialize iVIT-I NVIDIA Function

    Args:
        type (Literal[&quot;CLS&quot;, &quot;OBJ&quot;, &quot;SEG&quot;]): the model type
        params (dict): the model setting

    Raises:
        InvalidModelTypeError: get unexpected model type

    Returns:
        iModel: return model which inherit iModel
    """

    info = SERV_CONF['IDEV'].get_device_info( [ str(params['device']) ] )
    dev = int([ val['id'] for val in info.values() ][0])

    if type == CLS:
        from ivit_i.core.models import iClassification
        model = iClassification(
            model_path = params['model_path'],
            label_path = params['label_path'],
            device = dev,
            confidence_threshold = float(params['confidence_threshold']),
            topk = params.get('topk', 1),
            preproc_mode = "caffe"
        )
    elif type == OBJ:
        from ivit_i.core.models import iDetection
        model = iDetection(
            model_path = params["model_path"],
            label_path = params["label_path"],
            device = dev,
            anchors = params["anchors"],
            confidence_threshold = float(params["confidence_threshold"])
        )
    else:
        raise InvalidModelTypeError('Unexpect Type: {}'.format(type))

    return model

# Hailo
def ivit_i_hailo(type: Literal["CLS", "OBJ", "SEG"], params:dict) -> iModel:
    """Initialize iVIT-I Hailo Function

    Args:
        type (Literal[&quot;CLS&quot;, &quot;OBJ&quot;, &quot;SEG&quot;]): the model type
        params (dict): the model setting

    Raises:
        InvalidModelTypeError: get unexpected model type

    Returns:
        iModel: return model which inherit iModel
    """
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
            confidence_threshold = float(params["confidence_threshold"])
        )
    else:
        raise InvalidModelTypeError('Unexpect Type: {}'.format(type))
    
    return model

# Xilinx
def ivit_i_xlnx(type: Literal["CLS", "OBJ", "SEG"], params:dict) -> iModel:
    """Initialize iVIT-I Xilinx Function

    Args:
        type (Literal[&quot;CLS&quot;, &quot;OBJ&quot;, &quot;SEG&quot;]): the model type
        params (dict): the model setting

    Raises:
        InvalidModelTypeError: get unexpected model type

    Returns:
        iModel: return model which inherit iModel
    """
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
            confidence_threshold = float(params["confidence_threshold"])
        )
    else:
        raise InvalidModelTypeError('Unexpect Type: {}'.format(type))
    
    return model



def get_ivit_api(framework:Literal["openvino", "tensorrt", "jetson", "vitis-ai", "hailort"]) -> iModel:
    """Get iVIT-I API

    Args:
        framework (Literal[&quot;openvino&quot;, &quot;tensorrt&quot;, &quot;jetson&quot;, &quot;vitis): supported framework

    Returns:
        iModel: return target model which inherit with iModel
    """
    map = {
        "openvino": ivit_i_intel,
        "tensorrt": ivit_i_dgpu,
        "vitis-ai": ivit_i_xlnx,
        "hailort": ivit_i_hailo
    }

    return map[framework]

