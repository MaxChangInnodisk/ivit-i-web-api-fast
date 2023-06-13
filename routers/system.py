# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json, threading, time, sys, os, re
import logging as log
from fastapi import APIRouter,status
from fastapi.responses import Response, FileResponse
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


@sys_router.get("/files/{file_path}")
def download_file(file_path):
    try:
        file_path = os.path.join( SERV_CONF["EXPORT_DIR"], file_path)

        log.info('Start to download file: {}'.format(file_path))
        return FileResponse(path=file_path, filename=os.path.basename(file_path))
    except Exception as e:
        return http_msg(content=e, status_code=500)


@sys_router.get("/user/permit")
def change_user_permission():
    import subprocess as sb
    sb.run("chown 1000:1000 -R .", shell=True)
    return http_msg("done")

