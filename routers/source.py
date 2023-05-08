# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json, os, shutil, copy, cv2
import logging as log
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
# from typing_extensions import Annotated

from typing import List, Optional
from pydantic import BaseModel

from ..handlers.mesg_handler import http_msg
from ..handlers.io_handler import get_source_frame, add_source, get_src_info, create_source
from ..handlers.db_handler import select_data, insert_data, delete_data, update_data

# Router
source_router = APIRouter(tags=["source"])

# Format

class DelSourceFormat(BaseModel):
    uids: List[str]


class FrameFormat(BaseModel):
    height: int
    width: int

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
        src = create_source(source_uid=uid)
        frame = copy.deepcopy(src.frame) 
        ret, image = cv2.imencode(".jpeg", frame)
        buf = image.tobytes()
        src.release()
        return Response(    
            content = buf, 
            status_code = 200, media_type="image/jpeg" )
    except Exception as e:
        update_data(table='source', data={'status': 'error'}, condition=f"WHERE uid='{uid}'")
        return http_msg(content=e, status_code=500)
    
@source_router.post("/sources/{uid}/frame")
def get_frame_from_source_with_resolution(uid:str, data: Optional[FrameFormat]=None):
    try:
        
        src = create_source(source_uid=uid)
        frame = copy.deepcopy(src.frame) 
        if data and data.width and data.height:
            frame = cv2.resize( frame, (data.width, data.height))
        ret, image = cv2.imencode(".jpeg", frame)
        buffer = image.tobytes()
        src.release()
        return Response(    
            content = buffer, 
            status_code = 200, media_type="image/jpeg" )
    except Exception as e:
        update_data(table='source', data={'status': 'error'}, condition=f"WHERE uid='{uid}'")
        return http_msg(content=e, status_code=500)