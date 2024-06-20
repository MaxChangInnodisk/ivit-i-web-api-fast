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

from common import ICAP_CONF, SERV_CONF, WS_CONF, init_ivit_env, manager

init_ivit_env()

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
    length = len(web_url)
    log.info("-" * length)
    log.info("|" + " " * (length - 2) + "|")
    log.info(web_url)
    log.info("|" + " " * (length - 2) + "|")
    log.info("-" * length)


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
            "Unexpected request key, support key is {}".format(", ".join(SUP_KEYS))
        )

    # Parse keys
    return req[K_DATA], req[K_TYPE]


@app.websocket(f"{API_VERSION}/ws")
async def websocket_endpoint_task(ws: WebSocket):
    """
    WebSocket Router

    Support:
        1. Device Temperature
        2. Task Inference Result

    Request:
        {
            "type": < UID | TEMP | ERR | PROC >,
            "data": < task_uid | target_device_name: Union[] >
        }

    Response:
        {
            "type": < UID | TEMP | ERR | PROC >,
            "message": < error_message >,
            "data": < infer_result |  device_info >
        }

    """

    try:
        await manager.connect(ws)

        while True:
            # Check keys
            req = await ws.receive_json()
            req_data, req_type = parse_ws_req(req)

            # Get Data
            if req_type == UID:
                task_uid = req_data.upper()
                if task_uid not in WS_CONF:
                    await manager.send(
                        ws=ws,
                        message={
                            K_TYPE: ERR,
                            K_DATA: f"task uuid {task_uid} not in WS_CONF",
                        },
                    )
                    continue

                await manager.send(
                    ws=ws,
                    message={
                        K_TYPE: req_type,
                        K_DATA: get_pure_jsonify(WS_CONF.get(task_uid)),
                    },
                )
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
                await manager.send(
                    ws=ws,
                    message={K_TYPE: req_type, K_DATA: get_pure_jsonify(data)},
                )
                continue

            if req_type == PRO:
                proc_uid = req_data.upper()
                if proc_uid not in WS_CONF:
                    await manager.send(
                        ws=ws,
                        message={K_TYPE: ERR, K_DATA: "process uuid not in WS_CONF"},
                    )
                    continue
                # Send JSON Data
                await manager.send(
                    ws=ws,
                    message={
                        K_TYPE: req_type,
                        K_DATA: get_pure_jsonify(WS_CONF.get(proc_uid)),
                    },
                )

    # Disconnect
    except WebSocketDisconnect as e:
        log.warning(f"Disconnected: {e}")

    # Capture Error ( Exception )
    except Exception as e:
        await manager.broadcast(mesg_handler.ws_msg(type=ERR, content=e))

    finally:
        manager.disconnect(websocket=ws)


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

    uvicorn.run(
        "main:app",
        host=SERV_CONF["HOST"],
        port=int(SERV_CONF["PORT"]),
        workers=1,
        log_config="./uvicorn_logger.json",
    )
