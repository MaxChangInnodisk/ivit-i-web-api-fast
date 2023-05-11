# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json
import logging as log
from fastapi import APIRouter,status
from fastapi.responses import Response
from typing import Optional
from pydantic import BaseModel


from ..common import SERV_CONF
from ..handlers.mesg_handler import http_msg
from ..handlers.ivit_handler import iAPP_HANDLER
from ..handlers.app_handler import parse_app_data
from ..handlers.db_handler import select_data, insert_data


# Router
app_router = APIRouter( tags=["app"] )

# Helper Function

def get_app_info(uid: str=None):
    """ Get app Information from database """
    if uid == None:    
        data = select_data(table='app', data="*")
    else:
        data = select_data(table='app', data="*", condition=f'WHERE uid="{uid}"')
    ret = [ parse_app_data(app) for app in data ]
    return ret
        
# API

@app_router.get("/apps")
async def get_using_applicatino_list():

    try:
        ret = get_app_info()        
        return http_msg( content=ret, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


@app_router.get("/apps/support")
async def get_supported_application():

    try:
        APP='APP'
        if SERV_CONF.get(APP) is None:
            SERV_CONF[APP] = iAPP_HANDLER()
            SERV_CONF[APP].register_from_folder(SERV_CONF["APP_DIR"])
        sort_apps = SERV_CONF[APP].get_sort_apps()
        log.info(sort_apps)
        
        return http_msg( content=sort_apps, status_code=200)
    
    except Exception as e:
        return http_msg( content=e, status_code=500)
    

@app_router.get("/apps/{uid}")
async def get_target_application_information(uid:Optional[str]=None):

    try:
        ret = get_app_info(uid=uid)        
        return http_msg( content=ret, status_code=200 )

    except Exception as e:
        return http_msg( content=e, status_code=500 )


