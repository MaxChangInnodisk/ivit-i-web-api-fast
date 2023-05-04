# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .task import task_router
from .source import source_router
from .model import model_router
from .app import app_router
from .device import device_router
from .icap import icap_router
from .process import proc_router

routers = [
    task_router,
    source_router,
    model_router,
    app_router,
    device_router,
    icap_router,
    proc_router
]