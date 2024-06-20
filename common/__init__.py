# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from common.ivit_socket import manager

from .config import EVENT_CONF, ICAP_CONF, MODEL_CONF, RT_CONF, SERV_CONF, WS_CONF
from .env import get_ivit_logger_format, init_ivit_env, init_ivit_logger

init_ivit_env()
