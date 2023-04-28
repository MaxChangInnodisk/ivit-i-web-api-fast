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

from ..handlers import task_handler
from ..handlers.mesg_handler import http_msg
from ..handlers.db_handler import update_data

# --------------------------------------------------------------------
# Router
task_router = APIRouter()

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

class TaskAction(BaseModel):
    uid: str
    action: str
    data: Optional[TaskActionData]

class AddTaskFormat(BaseModel):
	name: str
	source_uid: str
	model_uid: str
	device: str
	model_setting: dict
	app_name: str
	app_setting: dict

class EditTaskFormat(BaseModel):
	uid: str
	name: str
	source_uid: str
	model_uid: str
	device: str
	model_setting: dict
	app_name: str
	app_setting: dict


class DelTaskFormat(BaseModel):
    uid: str

# --------------------------------------------------------------------
# API
@task_router.get("/task", tags=["task"])
async def get_task_list():
    """ Get All AI Task """
    ret = task_handler.get_task_info()
    return http_msg( content = ret, status_code = 200 )


@task_router.get("/task/exec", tags=["task"])
async def execute_task_usage():
    return http_msg( content=list(ACTION.keys()), status_code=200)


@task_router.get("/task/{uuid}", tags=["task"])
async def get_target_task_information(uuid: str):
    ret = task_handler.get_task_info(uid=uuid)
    return http_msg( content = ret, status_code = 200 )


@task_router.post("/task/exec", tags=["task"])
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
        update_data(table='task', data={'status':'error'}, condition=f"WHERE uid='{uid}'")
        
        return http_msg( content=e, status_code=500)


@task_router.post("/task", tags=["task"])
def add_task(add_data: AddTaskFormat):

    try:
        ret = task_handler.add_ai_task(add_data=add_data)
        return http_msg(content=ret)
    except Exception as e:
        return http_msg(content=e, status_code=500)


@task_router.delete("/task", tags=["task"])
def delete_task(del_data: DelTaskFormat):

    try:
        task_handler.del_ai_task(del_data.uid)
        return http_msg("Success")
    
    except Exception as e:
        return http_msg(content=e, status_code=500)


@task_router.put("/task", tags=["task"])
def edit_task(edit_data: EditTaskFormat):
    try:
        ret = task_handler.edit_ai_task(edit_data=edit_data)
        return http_msg(content=ret)
    except Exception as e:
        return http_msg(content=e, status_code=500)








