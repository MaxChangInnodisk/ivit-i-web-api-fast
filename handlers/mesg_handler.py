# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union, Literal
import json
import logging as log
from fastapi.responses import Response

try:
    from ..common import init_ivit_env
except:
    from common import init_ivit_env

from .ivit_handler import simple_exception, handle_exception

K_MESG  = "message"
K_CODE  = "status_code"
K_DATA  = "data"
K_TYPE  = "type"

# WebSocket
K_UID = "UID"
K_ERR = "ERROR"

def json_exception(content) -> dict:
    """ Return a iVIT Exception with JSON format """
    
    err_type, err_detail = simple_exception(content)
    
    # if not err_type in [ "ImageOpenError", "VideoOpenError", "RtspOpenError", "UsbCamOpenError" ]:
    #     err_type = "RuntimeError"
    
    return { 
        K_MESG: err_detail if isinstance(err_detail, str) else json.dumps(err_detail),
        K_TYPE: err_type 
    }


def http_msg_formatter(content, status_code:int=200):
    """ HTTP response handler """

    # Checking Input Type
    if not isinstance(status_code, int):
        raise TypeError(f"Status Code should be integer, but got {type(status_code)}")

    # Define Basic Format
    ret = {
        K_CODE: status_code,
        K_DATA: {},
        K_MESG: "",
        K_TYPE: ""
    }

    # If is Exception
    if isinstance(content, Exception):
        log.exception(content)
        # Update Message and Type
        ret.update(json_exception(content=content))
        
    # If not Exception, check input content is String or Object
    elif isinstance(content, str):
        ret[K_MESG] = content

    else:
        ret[K_DATA] = content
    
    return ret

def ws_msg(content:Union[str, dict], type:Literal["UID","ERROR","TEMP", "PROC"]) -> dict:
    """ Web Socket response handler """
    
    # Use Http Formatter
    ret = http_msg_formatter(content=content)

    # Update Type if not error
    if ret[K_TYPE] == "":
        ret[K_TYPE] = type

    # Remove status_code
    ret.pop(K_CODE, None)
    
    return ret

def http_msg(content, status_code:int=200, media_type:str="application/json"):
    """ HTTP response handler """

    ret = http_msg_formatter(content=content, status_code=status_code)
    
    return Response(    
        content = json.dumps(ret), 
        status_code = status_code, media_type=media_type )
