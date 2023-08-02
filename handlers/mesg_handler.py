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
    
    err_type, err_detail = simple_exception(error=content)
    
    # if not err_type in [ "ImageOpenError", "VideoOpenError", "RtspOpenError", "UsbCamOpenError" ]:
    #     err_type = "RuntimeError"
    
    return { 
        K_MESG: err_detail if isinstance(err_detail, str) else json.dumps(err_detail),
        K_TYPE: err_type 
    }

def http_msg_formatter(content: Union[dict, str, Exception], status_code:int=200) -> dict:
    """HTTP response handler

    Args:
        content (Union[dict, str, Exception]): _description_
        status_code (int, optional): _description_. Defaults to 200.

    Raises:
        TypeError: _description_

    Returns:
        dict: a dictionaray with `status_code`, `data`, `message`, `type`.

    Samples:
        ```python
        {
            status_code: status_code,
            data: {},
            message: "",
            type: ""
        }
        ```
    """
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

def ws_msg(content: Union[dict, str, Exception], type:Literal["UID", "ERROR", "TEMP", "PROC"]) -> dict:
    """Return a WebSocket Message

    Args:
        content (Union[dict, str, Exception]): message
        type (Literal[&quot;UID&quot;,&quot;ERROR&quot;,&quot;TEMP&quot;, &quot;PROC&quot;,&quot;EVENT&quot;]): websocket message type

    Returns:
        dict: websocket message

    Samples:
        ```
        {
            data: {},
            message: "",
            type: ""
        }
        ```
    """

    # Use Http Formatter
    ret = http_msg_formatter(content=content)

    # Update Type if not error
    if ret[K_TYPE] == "":
        ret[K_TYPE] = type

    # Remove status_code
    ret.pop(K_CODE, None)
    
    return ret

def http_msg(content: Union[dict, str, Exception], status_code:int=200, media_type:str="application/json") -> Response:
    """Return a HTTP Message

    Args:
        content (Union[dict, str, Exception]): message
        status_code (int, optional): the response status code. Defaults to 200.
        media_type (str, optional): the metdia type. Defaults to "application/json".

    Returns:
        Response: the response format from `fastapi`.

    Samples:
        ```python
        {
            status_code: status_code,
            data: {},
            message: "",
            type: ""
        }
        ```
    """
    ret = http_msg_formatter(content=content, status_code=status_code)
    
    return Response(
        content = json.dumps(ret), 
        status_code = status_code, media_type=media_type )
