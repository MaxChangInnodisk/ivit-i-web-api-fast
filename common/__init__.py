# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .env import (
    init_ivit_env,
    init_ivit_logger
)
from .config import (
    SERV_CONF, 
    MODEL_CONF, 
    ICAP_CONF, 
    RT_CONF, 
    WS_CONF,
    EVENT_CONF
)

from .ivit_socket import manager

init_ivit_env()
init_ivit_logger()