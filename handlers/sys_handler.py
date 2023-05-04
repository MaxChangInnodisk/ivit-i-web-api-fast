# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging as log
import subprocess as sp

def get_v4l2() -> tuple:
    """ Get V4L2 device list [] """
    ret_status, ret_message = True, []

    # Get V4L2 Camera
    command = sp.run("ls /dev/video*", shell=True, stdout=sp.PIPE, encoding='utf8')
    
    # 0 means success
    if command.returncode == 0:

        # Parse Each Camera to a list
        ret_message = command.stdout.strip().split('\n')
        log.debug("{}, {}".format(ret_message, type(ret_message)))
        
        # Check is failed_key in that information
        for msg in ret_message.copy():
            if int(msg.split("video")[-1])%2==1:
                # if N is even it's not available for opencv
                ret_message.remove(msg)
    
    # else not success
    else:
        ret_status  = False
        ret_message = "Camera not found"

    return ret_status, ret_message