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

# About FastAPI
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .utils import check_json, get_pure_jsonify
from .common import SERV_CONF, WS_CONF
from .samples import init_samples
from .handlers import (
    model_handler, 
    icap_handler, 
    app_handler, 
    db_handler
)

# About ICAP
import paho

# Routers
from .routers import routers

# Start up
app = FastAPI( root_path = SERV_CONF['ROOT'] )

# Resigter Router
for router in routers:
    app.include_router( router )

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():

    model_handler.init_db_model()    # Models
    app_handler.init_db_app()      # AppHandler
    icap_handler.init_icap()        # iCAP
    db_handler.reset_db()
    
@app.on_event("shutdown")
def shutdown_event():
    log.warning('Stop service ...')


@app.websocket("/ws")
async def websocket_endpoint_task(websocket: WebSocket):
    """
    WebSocket Router

    1. <UID>    : Result of Inference
    2. ERROR    : Runtime ErrorFormat
        a. format
            `{
            "uid": <uid>,
            "data": {}
            "message": "",
            "type": RuntimeError }
            `
    3. TEMP     : Temparature
    """
    UID = "UID"
    ERR = "ERROR"
    TEM = "TEMP"
    def init_uid(key:str, val=None):
        WS_CONF.update({ key: val})

    init_uid('WS', websocket)
            
    await websocket.accept()
    while True:
        # Receive Command
        key = await websocket.receive_text()
        
        # Get Return Data
        data = None
        if key in [ UID, ERR ]:
            data = WS_CONF.get(key)
        elif key == TEM:
            data = SERV_CONF['IDEV'].get_all_device()

        # Send Data
        if not data:
            await websocket.send_text("Got Unexpected key ({}) in WebSocket, Support is {}".format(key, ', '.join(WS_CONF.keys())))
            continue
        
        if isinstance(data, str):
            await websocket.send_text(data)
        else:
            log.debug('Send Data: {}'.format(data))
            await websocket.send_json(get_pure_jsonify(data))

        # Clear Data
        init_uid(key)

if __name__ == "__main__":

    # Initialize Database
    dp_path = SERV_CONF["DB_PATH"]
    framework = SERV_CONF["FRAMEWORK"]
    
    db_handler.init_tables(dp_path)
    if db_handler.check_db_is_empty(dp_path):
        db_handler.init_sqlite(dp_path)
        init_samples(framework=framework)

    # Fast API
    uvicorn.run(
        "service.main:app", 
        host = SERV_CONF["HOST"], 
        port = int(SERV_CONF["PORT"]),
        workers = 1,
        reload=True )


