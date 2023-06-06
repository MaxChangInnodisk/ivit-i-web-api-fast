# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import logging as log
from fastapi import APIRouter
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel
try:
    from ..common import SERV_CONF, ICAP_CONF
    from ..handlers.mesg_handler import http_msg
    from ..handlers import icap_handler
except:
    from common import SERV_CONF, ICAP_CONF
    from handlers.mesg_handler import http_msg
    from handlers import icap_handler

# Router

icap_router = APIRouter(tags=["icap"])

# Formatter

class TestFormat(BaseModel):
    type: str
    data: dict

class ResetFormat(BaseModel):
    ip: str
    port: str
    
# API

@icap_router.post("/icap/test")
async def test_attribute_and_telemetry(data:TestFormat):
    """ Test Attribute and Telemetry """
    try:
        type = data.type
        if 'attr' in type:
            SERV_CONF["ICAP"].send_attr(data.data)
        elif 'telem' in type:
            SERV_CONF["ICAP"].send_tele(data.data)
        else:
            raise NameError('Got Unexpected Type ...')
        
        return http_msg(content='Success', status_code=200)
    
    except Exception as e:
        return http_msg(content=e, status_code=500)


@icap_router.get("/icap/addr")
async def get_icap_address():
    return http_msg( { "ip" : str(ICAP_CONF["HOST"]), "port": str(ICAP_CONF["PORT"]) }, 200 )


@icap_router.post("/icap/addr")
async def set_icap_address(data: ResetFormat):
    try:
        ICAP_CONF["HOST"] = data.ip
        ICAP_CONF["PORT"] = data.port
        SERV_CONF["ICAP"].release()
        icap_handler.init_icap()
        return http_msg(content={ "ip" : str(ICAP_CONF["HOST"]), "port": str(ICAP_CONF["PORT"]) }, status_code=200)
    except Exception as e:
        SERV_CONF["ICAP"] = None
        return http_msg(content=e, status_code=500)

@icap_router.get("/icap/device/id")
async def get_device_id():
    return http_msg({"device_id": ICAP_CONF["DEVICE_ID"]}, 200 )

@icap_router.get("/icap/device/type")
async def get_device_id():
    return http_msg({"device_type": ICAP_CONF["DEVICE_TYPE"]}, 200 )
