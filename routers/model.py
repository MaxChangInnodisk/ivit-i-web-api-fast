# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json
import logging as log
from fastapi import APIRouter,File, Form, UploadFile
from fastapi.responses import Response
from typing import Optional, List
from pydantic import BaseModel

try:
    from ..handlers.mesg_handler import http_msg
    from ..handlers import model_handler
except:
    from handlers.mesg_handler import http_msg
    from handlers import model_handler

# Router
model_router = APIRouter( tags=["model"] )


# Format
class DelModelFormat(BaseModel):
    uids: List[str]


# API
@model_router.get("/models")
async def get_model_list():

    try:
        ret = model_handler.get_model_info()        
        return http_msg( content = ret, status_code = 200 )

    except Exception as e:
        return http_msg( content=e, status_code = 500 )
    
@model_router.get("/models/{uid}")
async def get_target_model_information(uid:Optional[str]):

    try:
        ret = model_handler.get_model_info(uid=uid)        
        return http_msg( content = ret, status_code = 200 )

    except Exception as e:
        return http_msg( content=e, status_code = 500 )

@model_router.delete("/models")
def delete_model(data: DelModelFormat):

    try:
        ret_data = { "success": [], "failure": []}
        for uid in data.uids:
            try:
                model_handler.delete_model(uid)
                ret_data["success"].append(uid)
            except:
                ret_data["failure"].append(uid)
        return http_msg( content = ret_data, status_code = 200 )

    except Exception as e:
        return http_msg( content=e, status_code = 500 )
    

@model_router.post("/models")
def add_model(
    file: Optional[UploadFile] = File(None),
    url: Optional[str]= Form(None),
):
    
    try:
        data = model_handler.add_model(file=file, url=url)
            
        return http_msg( 
            content = data, 
            status_code = 200 )

    except Exception as e:
        return http_msg( content=e, status_code = 500 )
    

