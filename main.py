# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
The iVIT-I Service Entry

1. Initialize Model
    * Update database
2. Initialize Application
    * Update database
3. Add Sample AI Task into database
    * Check Platform
4. Register iCAP
"""

# Basic
import json
import logging as log

# About FastAPI
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from common import ICAP_CONF, SERV_CONF, WS_CONF, manager
from handlers import (
    app_handler,
    db_handler,
    dev_handler,
    icap_handler,
    mesg_handler,
    model_handler,
)

# About ICAP
# Routers
from routers import routers
from samples import init_samples
from utils import get_pure_jsonify

UID = "UID"
ERR = "ERROR"
TEM = "TEMP"
PRO = "PROC"
K_DATA = "data"
K_TYPE = "type"
IDEV = "IDEV"
SUP_KEYS = [K_DATA, K_TYPE]

# Start up
app = FastAPI(root_path=SERV_CONF["ROOT"])


# Resigter Router
API_VERSION = "/v1"
for router in routers:
    app.include_router(router, prefix=API_VERSION)


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middle Ware
@app.middleware("http")
async def print_finish(request, call_next):
    response = await call_next(request)
    icap_handler.send_basic_attr()
    return response


def box_web_url():
    web_url = "|    WEBSITE --> http://{}:{}    |".format(
        SERV_CONF["HOST"], int(SERV_CONF["WEB_PORT"])
    )
    leng = len(web_url)
    log.info("-" * leng)
    log.info("|" + " " * (leng - 2) + "|")
    log.info(web_url)
    log.info("|" + " " * (leng - 2) + "|")
    log.info("-" * leng)


@app.on_event("startup")
def startup_event():
    try:
        model_handler.init_db_model()  # Models
    except Exception as e:
        log.warning(f"Init model error ... ({e})")

    try:
        app_handler.init_db_app()  # AppHandler
    except Exception as e:
        log.warning(f"Init application error ... ({e})")

    # Update iDev
    try:
        SERV_CONF["IDEV"] = dev_handler.iDeviceAsync()
    except Exception as e:
        log.warning(f"Init iDevice error ... ({e})")

    try:
        SERV_CONF["MQTT"] = mesg_handler.ServerMqttMessenger(
            "127.0.0.1", SERV_CONF["MQTT_PORT"]
        )

        icap_handler.init_icap(
            tb_url=ICAP_CONF["HOST"],
            tb_port=ICAP_CONF["PORT"],
            device_name=ICAP_CONF["DEVICE_NAME"],
        )  # iCAP
    except Exception as e:
        log.warning(f"Init MQTT error ... ({e})")

    db_handler.reset_db()
    log.info("iVIT-I Web Service Initialized !!!")
    box_web_url()


@app.on_event("shutdown")
def shutdown_event():
    db_handler.reset_db()
    log.warning("Stop service ...")


def parse_ws_req(req):
    # Check keys
    if sum([1 for sup in SUP_KEYS if sup in req.keys()]) < 2:
        raise KeyError(
            "Unexpect request key, support key is {}".format(", ".join(SUP_KEYS))
        )

    # Parse keys
    return req[K_DATA], req[K_TYPE]


@app.websocket("/{ver}/ws/temp")
async def websocket_temp(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            req = await ws.receive_json()

            # Check keys
            if sum([1 for sup in SUP_KEYS if sup in req.keys()]) < 2:
                raise KeyError(
                    "Unexpect request key, support key is {}".format(
                        ", ".join(SUP_KEYS)
                    )
                )

            # Parse keys
            req_data = req[K_DATA]
            req_type = req[K_TYPE]
            if isinstance(req_data, str):
                req_data = req_data.replace("'", '"')
                req_data = json.loads(req_data)
            data = (
                SERV_CONF[IDEV].get_device_info(req_data)
                if req_data != ""
                else SERV_CONF[IDEV].get_device_info()
            )
            await ws.send_json({K_TYPE: req_type, K_DATA: get_pure_jsonify(data)})

    # Disconnect
    except WebSocketDisconnect as e:
        print("Disconnected: ", e)

    # Capture Error ( Exception )
    except Exception as e:
        print(e)
        await ws.send_json(mesg_handler.ws_msg(type=ERR, content=e))


@app.websocket("/{ver}/ws/{task_uid}")
async def websocket_each_task(ws: WebSocket, ver: str, task_uid: str):
    # Connect WebSocket with Manager
    await manager.connect(ws=ws, uid=task_uid)

    try:
        while True:
            # Check keys
            req = await ws.receive_json()
            req_data, req_type = parse_ws_req(req)

            # Get Data
            req_data = req_data.upper()
            await manager.send(
                task_uid, get_pure_jsonify(WS_CONF.get(task_uid.upper()))
            )

            # Clear Data
            if req_type == UID:
                WS_CONF.update({req_data: None})

    # Disconnect
    except WebSocketDisconnect:
        manager.disconnect(ws=ws, uid=task_uid)
        await manager.broadcast(
            mesg_handler.ws_msg(type=UID, content=f"Task ({task_uid}:{id(ws)}) leave")
        )
    # Capture Error ( Exception )
    except Exception as e:
        await manager.broadcast(mesg_handler.ws_msg(type=ERR, content=e))


@app.websocket(f"{API_VERSION}/ws")
async def websocket_endpoint_task(ws: WebSocket):
    """
    WebSocket Router

    Support:
        1. Device Temparature
        2. Task Inference Result

    Request:
        {
            "type": < UID | TEMP | ERR | PROC >,
            "data": < task_uid | target_device_name: Uinon[] >
        }

    Response:
        {
            "type": < UID | TEMP | ERR | PROC >,
            "message": < error_message >,
            "data": < infer_result |  device_info >
        }

    """
    uid = str(id(ws))
    await manager.connect(ws, uid)

    while True:
        try:
            # Check keys
            req = await ws.receive_json()
            req_data, req_type = parse_ws_req(req)

            # Get Data
            data = None
            if req_type == UID:
                uid = req_data.upper()
                manager.register(ws, uid=uid)
                data = WS_CONF.get(uid)
                if data:
                    # Send JSON Data
                    await manager.send(
                        uid=uid,
                        message={K_TYPE: req_type, K_DATA: get_pure_jsonify(data)},
                    )
                    # Clear Data
                    WS_CONF.update({uid: None})

                continue

            if req_type == TEM:
                if isinstance(req_data, str):
                    req_data = req_data.replace("'", '"')
                    req_data = json.loads(req_data)
                data = (
                    SERV_CONF[IDEV].get_device_info(req_data)
                    if req_data != ""
                    else SERV_CONF[IDEV].get_device_info()
                )
                if data:
                    # await manager.broadcast(message=)
                    await manager.send(
                        uid=uid,
                        message={K_TYPE: req_type, K_DATA: get_pure_jsonify(data)},
                    )
                continue

            if req_type == PRO:
                uid = req_data.upper()
                try:
                    manager.register(ws, uid=uid)
                except BaseException:
                    pass
                data = WS_CONF.get(uid)
                if data:
                    # Send JSON Data
                    await manager.send(
                        uid=uid,
                        message={K_TYPE: req_type, K_DATA: get_pure_jsonify(data)},
                    )
                    # Clear Data
                    WS_CONF.update({uid: None})

        # Disconnect
        except WebSocketDisconnect as e:
            print("Disconnected: ", e)
            break

        except RuntimeError as e:
            print("Might disconnected", e)

        # Capture Error ( Exception )
        except Exception as e:
            await manager.broadcast(mesg_handler.ws_msg(type=ERR, content=e))


if __name__ == "__main__":
    # Initialize Database
    dp_path = SERV_CONF["DB_PATH"]
    framework = SERV_CONF["FRAMEWORK"]

    db_handler.init_tables(dp_path)
    if db_handler.is_db_empty(dp_path):
        db_handler.init_sqlite(dp_path)
        try:
            init_samples(framework=framework)
        except Exception as e:
            log.warning(f"Init sample error ... ({e})")

    # Fast API

    # uvicorn.run(
    #     "service.main:app",
    #     host = SERV_CONF["HOST"],
    #     port = int(SERV_CONF["PORT"]),
    #     workers = 1,
    #     reload=True )

    uvicorn.run(
        "main:app", host=SERV_CONF["HOST"], port=int(SERV_CONF["PORT"]), workers=1
    )
