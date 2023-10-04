# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json
import logging as log
from fastapi import APIRouter,status
from fastapi.responses import Response
from typing import Optional, Dict, List
from pydantic import BaseModel
import cv2
import os

try:
    from ..common import SERV_CONF, RT_CONF
    from ..handlers.mesg_handler import http_msg
    from ..handlers.db_handler import (
        select_data, insert_data, 
        parse_event_data, parse_task_data, 
        connect_db, close_db, db_to_list
    )
    from ..handlers import event_handler
    from ..handlers.app_handler import create_app
    from .source import frame2buf
except:
    from common import SERV_CONF, RT_CONF
    from handlers.mesg_handler import http_msg
    from handlers.db_handler import (
        select_data, insert_data, 
        parse_event_data, parse_task_data, 
        connect_db, close_db, db_to_list
    )
    from handlers import event_handler
    from handlers.app_handler import create_app
    from routers.source import frame2buf

# Router
event_router = APIRouter( tags=["event"] )

# --------------------------------------------------------------------
# Helper Function

# --------------------------------------------------------------------
# Request Body
class DelEventFormat(BaseModel):
    uids: List[str]

class getEventFormat(BaseModel):
    event_uid: Optional[str] = None,
    app_uid: Optional[str] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,

class ScreenShotFormat(BaseModel):
    timestamp: int
    draw_results: Optional[bool] = False
        
# --------------------------------------------------------------------
# API

@event_router.get("/events")
async def get_all_events():
    data = event_handler.get_all_events()
    return http_msg(content=data, status_code=200)

@event_router.post("/events")
async def get_target_events(data: getEventFormat):
    try:
        return http_msg(event_handler.get_events(data))

    except Exception as e:
    
        log.exception(e)
    
        return http_msg(content= e, status_code=500)


@event_router.delete("/events")
def get_events(del_data: DelEventFormat):

    try:
        ret_data = {
            'success': [],
            'failure': []
        }
        for uid in del_data.uids:
            try:
                event_handler.del_event(uid)
                
                ret_data["success"].append(uid)
            except:
                ret_data["failure"].append(uid)
        return http_msg(ret_data)
    
    except Exception as e:
        return http_msg(content=e, status_code=500)


@event_router.post("/events/screenshot")
def get_screenshot(data: ScreenShotFormat):
    """Get the screenshot of event"""
    try:
        frame = event_handler.get_event_screenshot(data.timestamp, data.draw_results)
        return Response(    content = frame2buf(frame=frame), 
                            status_code = 200, 
                            media_type="image/jpeg" )
    except Exception as e:
        return http_msg(    content=e, 
                            status_code=500)
