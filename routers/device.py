# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


# Basic
import logging as log
from fastapi import APIRouter

from ..common import SERV_CONF
from ..handlers.mesg_handler import http_msg

# Router
device_router = APIRouter()

@device_router.get("/platform", tags=["device"])
async def get_platform():
    return http_msg( content=SERV_CONF["PLATFORM"], status_code=200) 

@device_router.get("/device", tags=["device"])
async def get_device():
    return http_msg( content=SERV_CONF["IDEV"].get_all_device(), status_code=200)