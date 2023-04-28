# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json, os, shutil
import logging as log
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
# from typing_extensions import Annotated

from typing import List, Optional
from pydantic import BaseModel

from ..handlers.mesg_handler import http_msg
from ..handlers.io_handler import get_source_frame, add_source, get_src_info
from ..handlers.db_handler import select_data, insert_data, delete_data

# Router
source_router = APIRouter()

# Format
class SourceFormat(BaseModel):
    uid: str


# API
@source_router.get("/source", tags=["source"])
async def get_source_list():

    try:
        ret = get_src_info()        
        return http_msg( content=ret, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


@source_router.get("/source/{uid}", tags=["source"])
async def get_target_source(uid:Optional[str]=None):

    try:
        ret = get_src_info(uid=uid)        
        return http_msg( content=ret, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


@source_router.post("/source", tags=["source"])
async def add_new_source(
    files: Optional[List[UploadFile]] = File(None),
    input: Optional[str]= Form(None),
    option: Optional[dict]= Form({}) ):
    """ Add new source
    ---

    - Expected Format: 
        `{
            "file": <Image | Video> ,
            "input": <RTSP | V4L2>,
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


@source_router.delete("/source", tags=["source"])
def del_source(src_data: SourceFormat):
    try:
        delete_data(table='source', condition=f"WHERE uid='{src_data.uid}'")
        return http_msg(content='Success', status_code=200)
    except Exception as e:
        return http_msg(content=e, status_code=500)


@source_router.get("/source/{uid}/frame", tags=["source"])
def get_frame_from_source(uid:str):
    try:
        image = get_source_frame(uid)
        buf = image.tobytes()
        return Response(    
            content = buf, 
            status_code = 200, media_type="image/jpeg" )
    except Exception as e:
        return http_msg(content=e, status_code=500)