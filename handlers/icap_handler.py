# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import requests, json, logging, os, sys
import logging as log
import paho.mqtt.client as mqtt
from typing import Union

from ..common import SERV_CONF, ICAP_CONF
from ..utils import get_mac_address

from .mesg_handler import handle_exception

HEAD_MAP = {
    "json": {'Content-type': 'application/json'}
}

KEY_RESP_DATA = "data"
KEY_RESP_CODE = "status_code"

# Define Thingsboard return key
TB_KEY_NAME     = "name" 
TB_KEY_TYPE     = "type" 
TB_KEY_ALIAS    = "alias"
TB_KEY_MODEL    = "production_model"

TB_KEY_TIME         = "createdTime"
TB_KEY_TOKEN_TYPE   = "credentialsType"
TB_KEY_ID           = "id"
TB_KEY_TOKEN        = "accessToken"


def response_status(code):
    """ Return the response is success or not """
    return (str(code)[0] not in [ '4', '5' ])


def request_exception(exception, calling_api):
    """ Handle exception from sending request """

    RESP_EXCEPTION_MAP = {
        Exception: {
            KEY_RESP_DATA: f"Unxepected Error !!! ({handle_exception(exception)})",
            KEY_RESP_CODE: 400        
        },
        requests.Timeout: { 
            KEY_RESP_DATA: f"Request Time Out !!! ({calling_api})",
            KEY_RESP_CODE: 400 
        },
        requests.ConnectionError: { 
            KEY_RESP_DATA: f"Connect Error !!! ({calling_api})",
            KEY_RESP_CODE: 400 
        }
    }

    return ( False, RESP_EXCEPTION_MAP.get(type(exception)) )


def resp_to_json(resp):
    """ Parsing response and combine to JSON format """

    code = resp.status_code
    data = { KEY_RESP_CODE: code }

    # Parse from string
    try: 
        resp_data = json.loads(resp.text)
    except Exception as e: 
        resp_data = resp.text

    if type(resp_data) == str:
        logging.debug('Convert string response to json with key `data`')
        resp_data = { KEY_RESP_DATA: resp_data }
    
    # Merge data  
    data.update(resp_data)
    return response_status(code), data


def send_post_api(trg_url, data, h_type='json', timeout=10, stderr=True):
    """ Using request to simulate POST method """
    
    try:
        resp = requests.post(trg_url, data=json.dumps(data), headers=HEAD_MAP[h_type], timeout=timeout)
        return resp_to_json(resp)

    except Exception as e:
        return request_exception(exception=e, calling_api=trg_url) 


def send_get_api(trg_url, h_type='json', timeout=10):
    """ Using request to simulate GET method """

    try:
        resp = requests.get(trg_url, headers=HEAD_MAP[h_type], timeout=10)
        return resp_to_json(resp)
    
    except Exception as e:
        return request_exception(exception=e, calling_api=trg_url) 


def register_tb_device(tb_url):
    """ Register Thingsboard Device
    
    - Web API: http://10.204.16.110:3000/api/v1/devices
    - Method: POST
    - Data:
        - Type: JSON
        - Content: {
                "name"  : "ivit-i-{IP}",
                "type"  : "IVIT-I",
                "alias" : "ivit-i-{IP}"
            }
    - Response:
        - Type: JSON
        - Content: {
                "data": {
                    "createdTime": 1662976363031,
                    "credentialsType": "ACCESS_TOKEN",
                    "id": "a5636270-3280-11ed-a9c6-9146c0c923c4",
                    "accessToken": "auWZ5o6exyX9eWEmm7p3"
                }
            }
    """

    platform = SERV_CONF["PLATFORM"]
    if platform == "jetson":
        # Jetson device have to mapping to nvidia
        platform = "nvidia"
    
    dev_name = "iVIT-I"
    dev_type = "{}-{}".format(dev_name, get_mac_address()) 
    dev_alias = dev_type
    
    send_data = { 
        TB_KEY_NAME  : dev_name,
        TB_KEY_TYPE  : dev_type,
        TB_KEY_ALIAS : dev_alias,
        TB_KEY_MODEL : platform 
    }
    
    header = "http://"
    if ( not header in tb_url ): tb_url = header + tb_url

    timeout = 3
    # logging.warning("[ iCAP ] Register Thingsboard Device ... ( Time Out: {}s ) \n{}".format(timeout, send_data))
    logging.info('Register iCAP with: {}'.format(send_data))
    ret, data = send_post_api(tb_url, send_data, timeout=timeout, stderr=False)

    # Register failed
    if not ret:
        logging.warning("[ iCAP ] Register Thingsboard Device ... Failed !")
        logging.warning("Send Request: {}".format(send_data))    
        logging.warning("   - URL: {}".format(tb_url))
        logging.warning("   - TOKEN: {}".format( TB_KEY_TOKEN ))
        logging.warning("   - Response: {}".format(data))
        return False, data
    
    # Register sucess
    data = data["data"]
    receive_data = {
        "STATUS": True,
        "CREATE_TIME": data[TB_KEY_TIME],
        "DEVICE_ID": data[TB_KEY_ID],
        "ACCESS_TOKEN": data[TB_KEY_TOKEN]
    }

    # Update to ICAP_CONF
    ICAP_CONF.update(send_data)
    ICAP_CONF.update(receive_data)

    logging.info("[ iCAP ] Register Thingsboard Device ... Pass !")
    for key, val in receive_data.items():
        logging.info(f"  - {key}: {val}")

    return True, data


class ICAP_HANDLER():

    def __init__(self, host:str, port:Union[str, int], token:str ):
        self.client = mqtt.Client(client_id="", clean_session=True, userdata=None, transport="tcp")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish
        
        logging.info('Initialize MQTT ...')
        
        [ logging.info(f"   - {key}: {val}") for key, val in zip(["HOST", "PORT", "TOKEN"], [host, port, token])]
        
        self.client.username_pw_set(token)
        self.client.connect_async(str(host), int(port), keepalive=60, bind_address="")
    
    def start(self):
        self.client.loop_start()
    
    def stop(self):
        self.client.loop_stop(force=False)
        
    def __del__(self):
        self.client.disconnect()

    def on_connect(self, client, userdata, flags, rc):

        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        # client.subscribe("test/test")
        # Connect Success

        if rc == 0:
            
            logging.info('MQTT Connected successfully')

            # For Basic
            for topic in [ ICAP_CONF["TOPIC_REC_ATTR"] ]:
                client.subscribe(topic) # subscribe topic
                logging.info('  - Subscribed: {}'.format(topic))

            # For Receive Command From RPC, and Send Attribute
            for topic in [ ICAP_CONF["TOPIC_REC_RPC"], ICAP_CONF["TOPIC_SND_ATTR"] ]:
                _topic = topic + "+"
                client.subscribe(_topic)
                logging.info('  - Subcribe: {}'.format(_topic))

        # Connect Failed
        else: logging.error('MQTT Got Bad connection. Code:', rc)
        
        # Send Shared Attribute to iCAP
        # send_basic_attr()

    def on_message(self,client, userdata, msg):
        log.debug(msg.topic+" "+str(msg.payload))
        log.debug(msg.payload.decode())
        log.debug(type(msg.payload.decode()))

    def on_publish(self, client, userdata, result):
        print("data published to thingsboard \n")

    def send_data(self, data: dict, topic: str):
        self.client.publish(topic, json.dumps(data))

    def send_attr(self, data: dict, topic:str=ICAP_CONF["TOPIC_REC_ATTR"]):
        self.client.publish(topic, json.dumps(data))

    def send_tele(self, data: dict, topic:str=ICAP_CONF["TOPIC_SND_TEL"]):
        self.client.publish(topic, json.dumps(data))


def init_icap():
    """ Initialize iCAP """

    # Get Register URL
    register_url = "{}:{}{}".format(
        ICAP_CONF["HOST"], ICAP_CONF["PORT"], ICAP_CONF["API_REG_DEVICE"])
    
    # Try to Register
    flag, data = register_tb_device(register_url)
    
    # If success
    icap_handler = None

    if flag:
        icap_handler = ICAP_HANDLER( 
            host=ICAP_CONF["HOST"], 
            port=ICAP_CONF["MQTT_PORT"],
            token=ICAP_CONF["ACCESS_TOKEN"] )
        icap_handler.start()

    SERV_CONF.update({"ICAP":icap_handler})
    logging.info("Update ICAP Object into {}".format(SERV_CONF.get_name))
    
if __name__ == "__main__":

    register_url = "{}:{}{}".format(
        ICAP_CONF["HOST"], ICAP_CONF["PORT"], ICAP_CONF["API_REG_DEVICE"])
    
    register_tb_device(register_url)
    icap_handler = ICAP_HANDLER( 
        host=ICAP_CONF["HOST"], 
        port=ICAP_CONF["MQTT_PORT"],
        token=ICAP_CONF["ACCESS_TOKEN"] )
    icap_handler.start()

    while(True):
        data = input('Press Q to leave: ')

        if data.lower() == 'q':
            break