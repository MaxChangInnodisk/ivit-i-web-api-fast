# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# Basic
import json, threading, time, sys, os, re
import logging as log
from fastapi import APIRouter,File, Form, UploadFile
from fastapi.responses import Response, FileResponse
from typing import Optional, Dict, List
from pydantic import BaseModel


try:
    from ..common import SERV_CONF
    from ..handlers import task_handler
    from ..handlers.mesg_handler import http_msg, json_exception
    from ..handlers.db_handler import update_data
except:
    from common import SERV_CONF
    from handlers import task_handler
    from handlers.mesg_handler import http_msg, json_exception
    from handlers.db_handler import update_data

# --------------------------------------------------------------------
# Router
task_router = APIRouter(tags=["task"])

# --------------------------------------------------------------------
# Helper for task_action
ACTION = {
    'run': task_handler.run_ai_task,
    'stop': task_handler.stop_ai_task,
    'update': task_handler.update_ai_task
}


# --------------------------------------------------------------------
# Request Body

class TaskActionData(BaseModel):
    cv_display: Optional[bool]
    area: Optional[list]
    palette: Optional[dict]
    thres: Optional[float]
    draw_bbox: Optional[bool]
    draw_result: Optional[bool]
    draw_area: Optional[bool]
    draw_tracking: Optional[bool]
    draw_line: Optional[bool]
    


class TaskAction(BaseModel):
    uid: str
    action: str
    data: Optional[TaskActionData]


class AddTaskFormat(BaseModel):
	task_name: str
	source_uid: str
	model_uid: str
	device: str
	model_setting: dict
	app_name: str
	app_setting: dict


class EditTaskFormat(BaseModel):
	task_uid: str
	task_name: str
	source_uid: str
	model_uid: str
	device: str
	model_setting: dict
	app_name: str
	app_setting: dict


class DelTaskFormat(BaseModel):
    uids: List[str]


class ExportTaskFormat(BaseModel):
    uids: List[str]
    to_icap: bool


# --------------------------------------------------------------------
# API
@task_router.get("/tasks")
async def get_task_list():
    """ Get All AI Task """
    ret = task_handler.get_task_info()
    return http_msg( content = ret, status_code = 200 )


@task_router.get("/tasks/exec")
def execute_task_usage():
    return http_msg( content=list(ACTION.keys()), status_code=200)


@task_router.get("/tasks/{uid}")
async def get_target_task_information(uid: str):
    try:
        ret = task_handler.get_task_info(uid=uid)
        return http_msg( content = ret, status_code = 200 )
    except Exception as e:
        return http_msg( content=e, status_code=500)

@task_router.post("/tasks/exec")
async def execute_task(exec_data: TaskAction):
    
    # Parse Request
    uid = exec_data.uid
    action = exec_data.action
    data = exec_data.data

    log.info('Execute AI Task')
    log.info(f'  - uid: {uid}')
    log.info(f'  - action: {action}')
    log.info(f'  - data: {data}')

    try:

        func = ACTION[action]
        
        mesg = func(uid=uid, data=data)

        return http_msg( content = mesg, status_code = 200 )

    except Exception as e:
        log.exception(e)
        update_data(
            table='task', 
            data={
                'status':'error',
                'error': json_exception(e)},
            condition=f"WHERE uid='{uid}'")
        
        return http_msg( content=e, status_code=500)


@task_router.post("/tasks")
def add_task(add_data: AddTaskFormat):

    try:
    
        ret = task_handler.add_ai_task(add_data=add_data)
    
        code = 200 if ret['status']=='success' else 500
    
        return http_msg(content=ret, status_code=code)
    
    except Exception as e:
        return http_msg(content=e, status_code=500)


@task_router.delete("/tasks")
def delete_task(del_data: DelTaskFormat):

    try:
        ret_data = {
            'success': [],
            'failure': []
        }
        for uid in del_data.uids:
            try:
                task_handler.del_ai_task(uid)
                ret_data["success"].append(uid)
            except:
                ret_data["failure"].append(uid)
        return http_msg(ret_data)
    
    except Exception as e:
        return http_msg(content=e, status_code=500)


@task_router.put("/tasks")
def edit_task(edit_data: EditTaskFormat):
    try:
        ret = task_handler.edit_ai_task(edit_data=edit_data)
        return http_msg(content=ret)
    except Exception as e:
        return http_msg(content=e, status_code=500)


@task_router.post("/tasks/export")
def export_task(data: ExportTaskFormat):
    try:
        ret = task_handler.export_ai_task(export_uids=data.uids, to_icap=data.to_icap)
        return http_msg(content=ret)
    except Exception as e:
        return http_msg(content=e, status_code=500)

@task_router.post("/tasks/import")
def import_task(
    file: Optional[UploadFile] = File(None),
    url: Optional[str]= Form(None),
):
    try:
        data = task_handler.import_ai_task(file=file, url=url)
        return http_msg( 
            content = data, 
            status_code = 200 )

    except Exception as e:
        return http_msg( content=e, status_code = 500 )
    



