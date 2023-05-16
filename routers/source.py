# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json, os, shutil, copy, cv2
import logging as log
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
# from typing_extensions import Annotated

from typing import List, Optional, Literal
from pydantic import BaseModel

try:
    from ..handlers.mesg_handler import http_msg
    from ..handlers.io_handler import get_source_frame, add_source, get_src_info, create_source
    from ..handlers.db_handler import select_data, insert_data, delete_data, update_data
except:
    from handlers.mesg_handler import http_msg
    from handlers.io_handler import get_source_frame, add_source, get_src_info, create_source
    from handlers.db_handler import select_data, insert_data, delete_data, update_data

# Router
source_router = APIRouter(tags=["source"])

# Format

class DelSourceFormat(BaseModel):
    uids: List[str]


class FrameFormat(BaseModel):
    height: int
    width: int

# Helper

def frame2buf(frame:np.ndarray, format:Literal['.jpeg', '.png']='.jpeg')-> bytes:
    return cv2.imencode(format, frame)[1].tobytes()

# API
@source_router.get("/sources")
async def get_source_list():

    try:
        ret = get_src_info()        
        return http_msg( content=ret, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


@source_router.get("/sources/{uid}")
async def get_target_source(uid:Optional[str]=None):

    try:
        ret = get_src_info(uid=uid)        
        return http_msg( content=ret, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


@source_router.post("/sources")
async def add_new_source(
    files: Optional[List[UploadFile]] = File(None),
    input: Optional[str]= Form(None),
    option: Optional[dict]= Form({}) ):
    """ Add new source
    ---

    - Expected Format: 
        `{
            "files": < Image | Video | Images > ,
            "input": < RTSP | V4L2 >,
            "option": {}
        }`
    """
    try:
        data = add_source(
            files=files, 
            input=input, 
            option=option )
        
        return http_msg( content=data, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


@source_router.delete("/sources")
def del_source(src_data: DelSourceFormat):
    try:
        ret_data = {
            'success': [],
            'failure': []
        }
        for uid in src_data.uids:
            try:
                delete_data(table='source', condition=f"WHERE uid='{uid}'")
                ret_data["success"].append(uid)
            except:
                ret_data["failure"].append(uid)
        return http_msg(ret_data)

    except Exception as e:
        return http_msg(content=e, status_code=500)


@source_router.get("/sources/{uid}/frame")
def get_frame_from_source(uid:str):
    try:
        frame = get_source_frame(source_uid=uid)
        return Response(    
            content = frame2buf(frame=frame), 
            status_code = 200, media_type="image/jpeg" )
    except Exception as e:
        update_data(table='source', data={'status': 'error'}, condition=f"WHERE uid='{uid}'")
        return http_msg(content=e, status_code=500)
    
@source_router.post("/sources/{uid}/frame")
def get_frame_from_source_with_resolution(uid:str, data: Optional[FrameFormat]=None):
    try:
        frame = get_source_frame(source_uid=uid, resolution=[ data.height, data.width])
        return Response(    
            content = frame2buf(frame=frame), 
            status_code = 200, media_type="image/jpeg" )
    except Exception as e:
        update_data(table='source', data={'status': 'error'}, condition=f"WHERE uid='{uid}'")
        return http_msg(content=e, status_code=500)