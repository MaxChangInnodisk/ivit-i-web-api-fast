# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


# Basic
import logging as log

from fastapi import APIRouter

try:
    from ..common import SERV_CONF
    from ..handlers.mesg_handler import http_msg
except:
    from common import SERV_CONF
    from handlers.mesg_handler import http_msg

# Router
device_router = APIRouter(tags=["device"])


@device_router.get("/platform")
async def get_platform():
    return http_msg(content=SERV_CONF["PLATFORM"], status_code=200)


@device_router.get("/devices")
async def get_device():
    try:
        data = None
        for _ in range(2):
            try:
                data = SERV_CONF["IDEV"].get_device_info()
                break
            except Exception as e:
                log.exception(e)

        if data is None:
            raise RuntimeError("Could not get device information.")

        # Old Version
        # data = SERV_CONF["IDEV"].get_device_info()
        return http_msg(content=data, status_code=200)

    except Exception as e:
        log.exception(e)
        eee = RuntimeError("Could not get device information.")
        return http_msg(content=eee, status_code=500)
