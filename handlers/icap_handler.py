# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import requests, json, logging, os, sys, hashlib, time, wget, shutil
import logging as log
import paho.mqtt.client as mqtt
from typing import Union

from ..common import SERV_CONF, ICAP_CONF
from ..utils import get_mac_address, get_address

from .mesg_handler import handle_exception
from .model_handler import ModelDeployerWrapper, URL_DEPLOYER
from . import task_handler

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


# HANDLER

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
    
    def release(self):
        self.stop()
        self.client.disconnect()

    def __del__(self):
        self.release()

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
        self.send_attr(data = {
                "ivitUrl": get_address(),
                "ivitTask": task_handler.get_task_info()
        })

    def __attr_deploy_event(self, data):
        deploy_event = ICAP_DEPLOYER( data = data)
        deploy_event.start_deploy()

    def __attr_event(self, data):
        print("[ Attribute Event ]")
        keys = data.keys()
        if 'sw_description' in keys:    
            logging.warning('Detected url from iCAP, start to deploy ...')
            self.__attr_deploy_event(data)

    def __rpc_event(self, request_idx, data):
        print("[ RPC Event ({}) ]".format(request_idx))
        method  = data["method"].upper()
        params  = data["params"]
        web_api = params["api"]
        data    = params.get("data")

        # trg_url = "http://{}:{}{}".format("127.0.0.1", app.config['PORT'], web_api)
        trg_url = "http://{}:{}{}{}".format(
            "127.0.0.1", SERV_CONF["NGINX_PORT"], '/ivit', web_api)

        ret, resp = send_get_api(trg_url) if method.upper() == "GET" else send_post_api(trg_url, data)

        send_topic  = ICAP_CONF['TOPIC_SND_RPC']+f"{request_idx}"

        self.send_data(resp, send_topic)
        log_data, dat_limit = str(resp), 100
        log_data = log_data if len(log_data)<dat_limit else log_data[:dat_limit] + ' ...... '
        print("[iCAP] Topic: {}, Data: {}".format(send_topic, log_data))

    def on_message(self,client, userdata, msg):
        topic = msg.topic
        data = json.loads(msg.payload.decode())
        if ICAP_CONF["TOPIC_REC_RPC"] in topic:
            request_idx = topic.split('/')[-1]
            self.__rpc_event(request_idx, data)

        elif ICAP_CONF["TOPIC_REC_ATTR"] in topic:
            self.__attr_event(data)

    def on_publish(self, client, userdata, result):
        print("[MQTT] Data published to thingsboard.")
        pass

    def send_data(self, data: dict, topic: str):
        self.client.publish(topic, json.dumps(data))

    def send_attr(self, data: dict, topic:str=ICAP_CONF["TOPIC_REC_ATTR"]):
        self.client.publish(topic, json.dumps(data))

    def send_tele(self, data: dict, topic:str=ICAP_CONF["TOPIC_SND_TEL"]):
        self.client.publish(topic, json.dumps(data))


# DEPLOYER

class ICAP_DEPLOYER(URL_DEPLOYER):
    """ Deployer for iCAP Model """

    def __init__(self, data:dict) -> None:
        super().__init__(data["sw_url"])

        # Get Basic Data
        self.title          = data["sw_title"]
        self.ver            = data["sw_version"]
        self.tag            = data["sw_tag"]
        self.url            = data["sw_url"]
        self.checksum       = data["sw_checksum"]
        self.checksum_type  = data["sw_checksum_algorithm"]
        self.descr          = data["sw_description"]    
        self.package_id     = data.get("sw_package_id", "None")    
        
        # Get Description Data
        self.platform = self.descr.get("applyProductionModel")
        self.project_name   = self.descr["project_name"]
        self.file_name      = self.descr["file_id"]
        self.file_size      = self.descr["file_size"]
        self.model_type     = self.descr["model_type"]
        self.model_classes  = self.descr["model_classes"]
        
        # Update File Path
        self.file_path = os.path.join( SERV_CONF["MODEL_DIR"], f'{self.project_name}.zip')
        
        # Status
        self.is_finish = False

    def download_event(self):
        """ Download file via URL from iVIT-T """
        self.update_status(self.S_DOWN)
        self.file_name = wget.download( self.url,self.file_path, bar=self._download_progress_event)
        self.file_folder =  os.path.splitext( self.file_path )[0]
        SERV_CONF["MODEL"][self.uid]["name"] = os.path.basename(self.file_folder)

    def finished_event(self):
        super().finished_event()
        self.is_finish = True

    def push_mesg(self):
        """ Not only push websocket but also push to mqtt for iCAP ( Thingboard ) """

        assert not (SERV_CONF.get("ICAP") is None), "Make sure iCAP is already register"
        
        # WebSocket
        super().push_mesg()
        
        # MQTT
        icap_data = {
            "sw_state": SERV_CONF['MODEL'][self.uid]['status'],
            "sw_error": SERV_CONF['MODEL'][self.uid]['message'],
            "sw_package_id": self.package_id
        }
        
        if self.is_finish:
            icap_data.update({
                "current_sw_title": self.title,
                "current_sw_version": self.ver })

        SERV_CONF["ICAP"].send_tele(icap_data)

# Helper Function

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
    
    dev_type = "iVIT-I"
    dev_name = "{}-{}".format(dev_type, get_mac_address()) 
    dev_alias = dev_name
    
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
    else:
        raise RuntimeError('Register iCAP Failed')

def send_basic_attr():
    if ('ICAP' in SERV_CONF) and not (SERV_CONF['ICAP'] is None):
        SERV_CONF['ICAP'].send_attr(data=task_handler.get_task_info())

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