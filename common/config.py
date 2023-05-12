# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" iVIT-Params
a global params object to store anything we have to use.
"""

import json, collections
import logging as log
from typing import Any

from .env import init_ivit_env
from ivit_i.utils import iDevice


# Wrapper
class ConfigWrapper(collections.UserDict):
    """ Config Wrapper could help to build up config object with lower case """

    def __setitem__(self, key: Any, item: Any) -> None:
        super().__setitem__(key.upper(), item)

    def update(self, data):
        super().update(self.__uppercase(data))

    def __uppercase(self, data: dict) -> dict:
        return { k.upper(): v for k, v in data.items() }
    
    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key.upper())

    @property
    def get_name(self):
        return self.__class__.__name__

# Update SERV_CONF at first time
def update_service_config_at_first(config_path:str='ivit-i.json'):
    """ Initailize iVIT-I Config """
    
    global first_update_flag
    if not first_update_flag: return

    with open(config_path, 'r') as f:
        json_data = json.load(f)
    
    trg_conf = SERV_CONF
    
    for key, val in json_data.items():
        
        if not (key in [ "SERVICE", "MQTT", "ICAP" ]):
            trg_conf[key] = val
            log.info('({}) Update {}: {}'.format(trg_conf.get_name, key, val))
            continue

        for sub_key, sub_val in val.items():
            # Point to target config object
            if key == "ICAP": trg_conf = ICAP_CONF
            else: trg_conf = SERV_CONF
            # Modify key and value
            trg_conf[sub_key] = sub_val                        
            log.info('({}) Update {}: {}'.format( trg_conf.get_name, sub_key, sub_val))

    first_update_flag = False
    log.info('Initailized iVIT-I Config')

""" Service Object """
SERV_CONF = ConfigWrapper(
        ROOT = "/ivit",
        HOST = '127.0.0.1',
        PORT = '819',
        ORIGINS = "*",
        DB_PATH = 'ivit_i.sqlite',
        APP_DIR = 'apps',
        DATA_DIR = 'data',
        MODEL_DIR = 'model',
        NGINX_PORT = "6632",
        RTSP_PORT = "8554",
        WEB_PORT = "4999",
        IDEV = iDevice()
)


""" Model Object """
MODEL_CONF = ConfigWrapper(
    
    # Platform
    NV = 'nvidia',
    JETSON = 'jetson',
    INTEL = 'intel',
    XLNX = 'xilinx',
    
    # Model Type,
    CLS = 'CLS',
    OBJ = 'OBJ',
    SEG = 'SEG',
    
    # Define extension for ZIP file form iVIT-T,
    DARK_LABEL_EXT = ".txt",
    CLS_LABEL_EXT = ".txt",        # txt is the category list,
    DARK_JSON_EXT  = ".json",       # json is for basic information like input_shape, preprocess,
    CLS_JSON_EXT = ".json",
    DARK_MODEL_EXT  = ".weights",
    ONNX_MODEL_EXT  = ".onnx",
    DARK_CFG_EXT    = ".cfg",
    TRT_MODEL_EXT   = ".trt",
    XLNX_MODEL_EXT  = ".xmodel",
    IR_MODEL_EXT    = ".xml",
    IR_MODEL_EXTS   = [ ".bin", ".mapping", ".xml" ],
)


""" iCAP Object """
ICAP_CONF = ConfigWrapper(
    HOST = "10.204.16.115",
    PORT = "3000",
    API_REG_DEVICE  = "/api/v1/devices",
    TOPIC_REC_RPC   = "v1/devices/me/rpc/request/",          # +
    TOPIC_SND_RPC   = "v1/devices/me/rpc/response/",         # {n}
    TOPIC_REC_ATTR  = "v1/devices/me/attributes",
    TOPIC_SND_ATTR  = "v1/devices/me/attributes/response/",  # +
    TOPIC_SND_TEL   = "v1/devices/me/telemetry",

    MQTT_PORT = "1883",
    STATUS = False,
    CREATE_TIME = "",
    DEVICE_ID = "",
    ACCESS_TOKEN = ""
)


""" Runtime Object to store each AI Task ( Thread Ojbect ) and Source ( Ojbect ) """
RT_CONF = ConfigWrapper(
    SRC = {}
)


""" WebSocket Object to store each WebSocket Object"""
WS_CONF = ConfigWrapper()

# Update config at first time
first_update_flag = True
update_service_config_at_first()
