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


# Helper 
def check_icap():
    
    if SERV_CONF.get('ICAP') is None:
        raise RuntimeError('iCAP is not register !! ')


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
    
    is_reg = SERV_CONF.get("ICAP")
    try:
        if is_reg is not None:
            log.warning('Re-register iCAP ')
            SERV_CONF["ICAP"].release()
        
        flag, data = icap_handler.init_icap(
            tb_url= data.ip,
            tb_port= data.port,
            device_name = ICAP_CONF.get("DEVICE_NAME", "")
        )
        if not flag:
            return http_msg(content=data, status_code=data["status_code"])
        
        return http_msg(content={ "ip" : str(ICAP_CONF["HOST"]), "port": str(ICAP_CONF["PORT"]) }, status_code=200)
    
    except Exception as e:
        log.exception(e)
        return http_msg(content=e, status_code=500)
    

@icap_router.get("/icap/device/id")
async def get_device_id():
    try:
        check_icap()
        return http_msg({"device_id": ICAP_CONF["DEVICE_ID"]}, 200 )
    except Exception as e:
        return http_msg(e, 500)

@icap_router.get("/icap/device/type")
async def get_device_id():
    try:
        check_icap()
        return http_msg({"device_type": ICAP_CONF["DEVICE_TYPE"]}, 200 )
    except Exception as e:
        return http_msg(e, 500)

@icap_router.get("/icap/device/name")
async def get_device_id():
    try:
        check_icap()
        return http_msg({"device_type": ICAP_CONF["DEVICE_NAME"]}, 200 )
    except Exception as e:
        return http_msg(e, 500)
