# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys, os
from ivit_i.common import ivit_logger

def init_ivit_env():
    
    ws = '/workspace'
    if ws in sys.path:
        return
    print('Initialize iVIT Environment')
    sys.path.append(ws)

def init_ivit_logger():
    pass
#     ivit_logger()


