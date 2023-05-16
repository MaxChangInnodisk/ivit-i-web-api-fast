# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json, threading, time, sys, os, re
import logging as log
from fastapi import APIRouter,status
from fastapi.responses import Response
from typing import Optional, Dict, List
from pydantic import BaseModel

try:
    from ..common import SERV_CONF
    from ..handlers import sys_handler
    from ..handlers.mesg_handler import http_msg
except:
    from common import SERV_CONF
    from handlers import sys_handler
    from handlers.mesg_handler import http_msg

# --------------------------------------------------------------------
# Router
sys_router = APIRouter(tags=["system"])

@sys_router.get('/v4l2')
def get_v4l2_device_list():
    try:
        ret, mesg = sys_handler.get_v4l2()
        if not ret: raise RuntimeError(mesg)
        return http_msg(content=mesg)
    
    except Exception as e:
        return http_msg(content=e, status_code=500)
