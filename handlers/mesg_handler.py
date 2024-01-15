# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import Union, Literal, Optional
import json
import logging as log
from fastapi.responses import Response
import paho.mqtt.client as mqtt
from functools import wraps
import asyncio

try:
    from ..common import init_ivit_env, SERV_CONF, RT_CONF, WS_CONF, EVENT_CONF, manager
    from .ivit_handler import simple_exception, handle_exception

except:
    from common import init_ivit_env, SERV_CONF, RT_CONF, WS_CONF, EVENT_CONF, manager
    from handlers.ivit_handler import simple_exception, handle_exception

# from common import init_ivit_env, SERV_CONF, RT_CONF, WS_CONF, EVENT_CONF
# from handlers.ivit_handler import simple_exception, handle_exception


K_MESG = "message"
K_CODE = "status_code"
K_DATA = "data"
K_TYPE = "type"

# WebSocket
K_UID = "UID"
K_ERR = "ERROR"


def json_exception(content) -> dict:
    """ Return a iVIT Exception with JSON format """

    err_type, err_detail = simple_exception(error=content)

    # if not err_type in [ "ImageOpenError", "VideoOpenError", "RtspOpenError", "UsbCamOpenError" ]:
    #     err_type = "RuntimeError"

    return {
        K_MESG: err_detail if isinstance(err_detail, str) else json.dumps(err_detail),
        K_TYPE: err_type
    }


def http_msg_formatter(content: Union[dict, str, Exception], status_code: int = 200) -> dict:
    """HTTP response handler

    Args:
        content (Union[dict, str, Exception]): _description_
        status_code (int, optional): _description_. Defaults to 200.

    Raises:
        TypeError: _description_

    Returns:
        dict: a dictionaray with `status_code`, `data`, `message`, `type`.

    Samples:
        ```python
        {
            status_code: status_code,
            data: {},
            message: "",
            type: ""
        }
        ```
    """
    # Checking Input Type
    if not isinstance(status_code, int):
        raise TypeError(
            f"Status Code should be integer, but got {type(status_code)}")

    # Define Basic Format
    ret = {
        K_CODE: status_code,
        K_DATA: {},
        K_MESG: "",
        K_TYPE: ""
    }

    # If is Exception
    if isinstance(content, Exception):
        log.exception(content)
        # Update Message and Type
        ret.update(json_exception(content=content))

    # If not Exception, check input content is String or Object
    elif isinstance(content, str):
        ret[K_MESG] = content

    else:
        ret[K_DATA] = content

    return ret


def ws_msg(content: Union[dict, str, Exception], type: Literal["UID", "ERROR", "TEMP", "PROC"]) -> dict:
    """Return a WebSocket Message

    Args:
        content (Union[dict, str, Exception]): message
        type (Literal[&quot;UID&quot;,&quot;ERROR&quot;,&quot;TEMP&quot;, &quot;PROC&quot;,&quot;EVENT&quot;]): websocket message type

    Returns:
        dict: websocket message

    Samples:
        ```
        {
            data: {},
            message: "",
            type: ""
        }
        ```
    """

    # Use Http Formatter
    ret = http_msg_formatter(content=content)

    # Update Type if not error
    if ret[K_TYPE] == "":
        ret[K_TYPE] = type

    # Remove status_code
    ret.pop(K_CODE, None)

    return ret


def http_msg(content: Union[dict, str, Exception], status_code: int = 200, media_type: str = "application/json") -> Response:
    """Return a HTTP Message

    Args:
        content (Union[dict, str, Exception]): message
        status_code (int, optional): the response status code. Defaults to 200.
        media_type (str, optional): the metdia type. Defaults to "application/json".

    Returns:
        Response: the response format from `fastapi`.

    Samples:
        ```python
        {
            status_code: status_code,
            data: {},
            message: "",
            type: ""
        }
        ```
    """
    ret = http_msg_formatter(content=content, status_code=status_code)

    return Response(
        content=json.dumps(ret),
        status_code=status_code, media_type=media_type)

# ---------------------------------------------------------------

# ---------------------------------------------------------------


class Messenger:
    """Handle message"""

    UID = "UID"
    ERR = "ERROR"
    TEM = "TEMP"
    PRO = "PROC"

    def write(self):
        pass


class MqttMessenger(Messenger):

    STOP = "stop"
    RUN = "run"
    ERR = "error"

    def __init__(self, broker_address: str, broker_port: str, client_id: str = ""):
        self.broker_address = broker_address
        self.broker_port = int(broker_port)
        self.client_id = client_id

        self.status = self.STOP

        self.client = mqtt.Client(self.client_id, clean_session=True)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        # self.client.on_publish = self.on_publish

    def _check_status(func):

        def wrap(self, *args, **kwargs):
            if self.status == self.STOP:
                print('is stop')
                return None
            return func(*args, **kwargs)
        return wrap

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info("Connected to MQTT broker")
            self.status = self.RUN
        else:
            log.info(f"Connection failed with code {rc}")
            self.status = self.ERR

    def on_disconnect(self, client, userdata,  rc):
        log.warning('Disconnected')

    def on_message(self, client, userdata, message):
        print(message.payload.decode("utf-8"))

    def connect(self):
        self.client.connect(self.broker_address, self.broker_port)
        self.client.loop_start()
        log.info('Connected MQTT Broker {}:{}'.format(
            self.broker_address, self.broker_port))

    def subscribe(self, topic):
        self.client.subscribe(topic)
        print('Subscribed Topic: {}'.format(topic))

    def publish(self, topic, message):

        if not isinstance(message, str):
            message = json.dumps(message)

        self.client.publish(topic, message)

    def set_message_callback(self, callback):
        self.client.on_message = callback

    def start(self):
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


class ServerMqttMessenger(MqttMessenger):

    EVENT = "events"
    RESULT = "results"

    def __init__(self, broker_address: str, broker_port: str, client_id: str = ""):
        super().__init__(broker_address, broker_port, client_id)

        self.connect()
        for topic in [self.EVENT]:
            self.subscribe(topic)
            self.publish(self.EVENT, 'iVIT Registered !')

    def subscribe_event(self, uid: str) -> None:
        assert uid != "", f"Subscribe event have to give it a uid."
        self.subscribe(self.get_event_topic(uid))

    def get_event_topic(self, uid: str = "") -> str:
        if uid == "":
            return self.EVENT
        return f"{self.EVENT}/{uid}"


class WebSocketMessenger(Messenger):

    def write(self, content, type, uid: Optional[str] = None):
        """ Push message """
        data = ws_msg(type=type, content=content)

        if uid:
            asyncio.run(manager.send(uid=uid.upper(), message=data))

        else:
            asyncio.run(manager.broadcast(data))
            asyncio.sleep(0)
            

class TaskMessenger(WebSocketMessenger):

    def push_mesg(self):
        """ Push message

        - Workflow
            1. Print Status
            2. If WebSocket exists then push message via `WS_CONF["WS"].send_json()`
        """
        try:
            self.write(
                type=self.PRO,
                content=SERV_CONF[self.PRO])

        except Exception as e:
            log.warning('WebSocket send error ... ', e)


if __name__ == "__main__":
    import time
    mc = ServerMqttMessenger(broker_address='127.0.0.1', broker_port='6683')
    TOPIC = 'test/topic'
    mc.subscribe(TOPIC)

    while (True):
        time.sleep(1)
        mc.publish(TOPIC, 'test')
        print('test')

    pass
