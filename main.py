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
import logging as log
import json

# About FastAPI
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from utils import check_json, get_pure_jsonify
from common import SERV_CONF, WS_CONF
from samples import init_samples
from handlers import (
    model_handler, 
    icap_handler, 
    app_handler, 
    db_handler
)
from handlers.mesg_handler import ws_msg

# About ICAP
import paho

# Routers
from routers import routers


# Start up
app = FastAPI( root_path = SERV_CONF['ROOT'] )


# Resigter Router
API_VERSION = '/v1'
for router in routers:
    app.include_router( router, prefix=API_VERSION )


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


@app.on_event("startup")
def startup_event():

    model_handler.init_db_model()    # Models
    app_handler.init_db_app()      # AppHandler
    icap_handler.init_icap()        # iCAP
    db_handler.reset_db()
    log.info('iVIT-I Web Service Initialized !!!')

@app.on_event("shutdown")
def shutdown_event():
    log.warning('Stop service ...')


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

    # Key
    UID = "UID"
    ERR = "ERROR"
    TEM = "TEMP"
    PRO = "PROC"

    # Data Key
    K_DATA = "data"
    K_TYPE = "type"

    # Other Key
    IDEV = "IDEV"

    SUP_KEYS = [ K_DATA, K_TYPE ]

    def init_uid(key:str, val=None):
        WS_CONF.update({ key: val})

    init_uid("WS", ws)
            
    await ws.accept()

    while True:

        try:
            
            req = await ws.receive_json()

            # Check keys
            if sum([ 1 for sup in SUP_KEYS if sup in req.keys() ]) < 2:
                raise KeyError('Unexpect request key, support key is {}'.format(
                    ', '.join(SUP_KEYS)))

            # Parse keys
            req_data = req[K_DATA]
            req_type = req[K_TYPE]

            # Get Data
            data = None
            if req_type == TEM:
                if isinstance(req_data, str):
                    req_data = req_data.replace("'", '"')
                    req_data = json.loads(req_data)
                data = SERV_CONF[IDEV].get_device_info(req_data) \
                    if req_data != "" else \
                        SERV_CONF[IDEV].get_device_info()
                    
            elif req_type == UID:
                data = WS_CONF.get(req_data.upper())

            # Invalid Key
            if not data:
                if req_type == UID:
                    sup_keys = list(WS_CONF.keys())
                    sup_keys.remove('WS')
                    raise KeyError("Got Invalid AI Task UID, Support is [ {} ]".format(
                        ', '.join( sup_keys ) ) )
                else:
                    raise KeyError("Got Unexpect Error")
            
            # Send JSON Data
            await ws.send_json({
                K_TYPE: req_type,
                K_DATA: get_pure_jsonify(data) 
            })

            # Clear Data
            if req_type == UID:
                init_uid(req_data)

        # Capture Error ( Exception )
        except Exception as e:
            await ws.send_json(
                ws_msg( type=ERR, content=e ))


if __name__ == "__main__":

    # Initialize Database
    dp_path = SERV_CONF["DB_PATH"]
    framework = SERV_CONF["FRAMEWORK"]
    
    db_handler.init_tables(dp_path)
    if db_handler.is_db_empty(dp_path):
        db_handler.init_sqlite(dp_path)
        init_samples(framework=framework)

    # Fast API

    # uvicorn.run(
    #     "service.main:app", 
    #     host = SERV_CONF["HOST"], 
    #     port = int(SERV_CONF["PORT"]),
    #     workers = 1,
    #     reload=True )

    uvicorn.run(
        "main:app", 
        host = SERV_CONF["HOST"], 
        port = int(SERV_CONF["PORT"]),
        workers = 1 )
