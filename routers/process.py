# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


# Basic
from fastapi import APIRouter

try:
    from ..common import SERV_CONF
    from ..handlers.mesg_handler import http_msg
except:
    from common import SERV_CONF
    from handlers.mesg_handler import http_msg

# --------------------------------------------------------------------
# Router
proc_router = APIRouter(tags=["process"])


# Helper


def check_proc():
    if SERV_CONF.get("PROC") is None:
        SERV_CONF.update({"PROC": {}})


# API
@proc_router.get("/process")
async def get_model_proc_list():
    try:
        check_proc()
        return http_msg(content=SERV_CONF["PROC"], status_code=200)

    except Exception as e:
        return http_msg(content=e, status_code=500)


@proc_router.get("/process/{uid}")
async def get_model_proc_list(uid: str):
    try:
        check_proc()
        return http_msg(content=SERV_CONF["PROC"].get(uid), status_code=200)

    except Exception as e:
        return http_msg(content=e, status_code=500)


@proc_router.delete("/process/{uid}")
async def remove_model_proc_list(uid: str):
    try:
        check_proc()
        SERV_CONF["PROC"].pop(uid)
        mesg = f"Delete the process uid: {uid}"
        return http_msg(content=mesg, status_code=200)

    except Exception:
        mesg = "Not found process"
        return http_msg(content=mesg, status_code=500)


@proc_router.delete("/process")
async def remove_model_proc_list():
    try:
        check_proc()
        SERV_CONF["PROC"] = {}
        return http_msg(content="Clear all process", status_code=200)

    except Exception:
        mesg = "Not found process"
        return http_msg(content=mesg, status_code=500)
