# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import requests, json, os, sys, hashlib, time, wget, shutil
import logging as log
import paho.mqtt.client as mqtt
from typing import Union, List

try:
    from ..common import SERV_CONF, ICAP_CONF
    from ..utils import get_mac_address, get_address, gen_uid

except:
    from common import SERV_CONF, ICAP_CONF
    from utils import get_mac_address, get_address, gen_uid

from .mesg_handler import handle_exception, http_msg_formatter
from .model_handler import ModelDeployerWrapper, MODEL_URL_DEPLOYER
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

# Helper

def dict_printer(title:str, data:dict, content_maximum:int = 50, level:str='info'):
    
    log_wrapper = {
        'debug': log.debug,
        'info': log.info,
        'warn': log.warning,
        'warning': log.warning,
    }

    _logger = log_wrapper[level]
    
    _logger("[ iCAP ] {}".format(title))
    
    for key, val in data.items():
        val = str(val)
        _logger("    - {}: {}".format(
            key, val if len(val)<=content_maximum else val[:content_maximum]+' ... '
        ))
    

# --------------------------------------------------------
# HANDLER

class ICAP_HANDLER():

    def __init__(self, host:str, port:Union[str, int], token:str ):
        self.client = mqtt.Client(client_id="", clean_session=True, userdata=None, transport="tcp")
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_publish = self.on_publish

        # Setup Password
        self.client.username_pw_set(token)

        # Log out
        dict_printer(   
            title='Init iCAP Handler:', 
            data={ "host": host,
                   "port": port,
                   "token ( pw )": token } )
        
        # self.client.connect(str(host), int(port), keepalive=60, bind_address="")

        self.client.connect_async(str(host), int(port), keepalive=60, bind_address="")

    # Basic

    def start(self):
        self.client.loop_start()
    
    def stop(self):
        self.client.loop_stop(force=False)
    
    def release(self):
        self.stop()
        self.client.disconnect()

    def __del__(self):
        self.release()

    # Private Event

    def _attr_deploy_event(self, data):
        """ Deployment Event """
        deploy_event = ICAP_DEPLOYER( data = data)
        deploy_event.start_deploy()

    def _attr_event(self, data):
        """ Attribute Event """
        if not data.get('ota_trigger', 0):
            log.warning('OTA Not Trigger !')
            return
            
        if 'sw_description' in data:
            if 'sw_url' in data:
                log.warning('Detected url from iCAP, start to deploy ...')
                self._attr_deploy_event(data)
            else:
                log.warning(f'The second time of iCAP message ...')    
        else:
            log.warning('Unexpected key ... {}'.format(', '.join(data.keys())))

    def _stream_behaviour(self, exec_data: dict) -> tuple:
        """Execute WebRTC streaming Add and Delete via ACTION

        Args:
            exec_data (dict): the data after exec.

        Returns:
            tuple: return code and response
        """
        rtc_ret, rtc_resp = 0, ""

        uid, action = exec_data["uid"], exec_data["action"]
        
        if action == "run":
            # Call stream add
            log.info('Add stream')
            rtc_url, rtc_data = f"http://127.0.0.1:8083/stream/{uid}/add", {
                "name":uid,
                "channels":{
                    "0":{
                        "name":"ch1",
                        "url":f"rtsp://127.0.0.1:8554/{uid}",
                        "on_demand":False,
                        "debug":False,
                        "status":0
                }}}

            rtc_ret, rtc_resp = send_post_api(rtc_url, rtc_data)

        elif action == "stop":
            # Call stream delete
            log.info('Delete stream')
            rtc_url = f"http://127.0.0.1:8083/stream/{uid}/delete"

            rtc_ret, rtc_resp = send_get_api(rtc_url)
        
        return rtc_ret, rtc_resp

    def _rpc_event(self, request_idx, data):
        """ Receive RPC Event """

        send_topic  = ICAP_CONF['TOPIC_SND_RPC']+f"{request_idx}"
        resp = None

        try:
            method  = data["method"].upper()
            web_api = data["params"].get("api")
            data    = data["params"].get("data")

            # Check 
            if web_api is None:
                raise KeyError("The RPC command must have 'api' key.")

            # Redirect URL
            trg_url = "http://{}:{}{}{}".format(
                "127.0.0.1", SERV_CONF["NGINX_PORT"], '/ivit', web_api)

            # Send Request
            if method == "GET":
                ret, resp = send_get_api(trg_url)  
            elif method == "POST":

                ret, resp = send_post_api(trg_url, data)

                if "exec" in trg_url: 
                    self._stream_behaviour(data)

            else:
                raise KeyError("Invalid method, sopported is 'GET', 'POST', 'PUT', 'DELETE'")

        except Exception as e:
            resp = http_msg_formatter(e, status_code=500)

        finally:
            self.send_data(resp, send_topic)

    # MQTT Event

    def on_disconnect(self, client, userdata, rc):
        log.warning('MQTT Closed!')

    def on_connect(self, client, userdata, flags, rc):
        """Connect to MQTT Method

        Workflow
            Connect to MQTT
            Send attributes with ivitUrl and ivitTask if success
        """

        if rc == 0:
            log.info('ICAP_HANDLER Connected successfully')
            
            # Define Topics
            topics = {
                "Send Attributes": ICAP_CONF["TOPIC_SND_ATTR"],
                "Receive Attributes": ICAP_CONF["TOPIC_REC_ATTR"]+'+',
                "Receive RPC Command": ICAP_CONF["TOPIC_REC_RPC"]+'+'
            }

            # Subscribe topic
            [ client.subscribe(t) for t in topics.values() ]

            dict_printer(title='Subscribe Topics:', data=topics) 

            # Send Shared Attribute to iCAP
            host_addr = get_address()
            if not (SERV_CONF["WEB_PORT"] is None):
                host_addr = host_addr + f':{SERV_CONF["WEB_PORT"]}'
                log.warning('Website address is : {}'.format(host_addr))

            # basic_attr = {
            #         "ivitUrl": host_addr,
            #         "ivitTask": task_handler.get_task_info()
            # }
            basic_attr = {
                    "ivitUrl": host_addr,
                    "ivitTask": get_icap_task_info()
            }
            self.send_attr(data = basic_attr)
            
            dict_printer(
                title='Send basic attributes:', 
                data=basic_attr)

        # Connect Failed
        else: log.error('MQTT Got Bad connection. Code:', rc)
     
    def on_message(self,client, userdata, msg):
        
        # Parse Data
        topic = msg.topic
        mesg = msg.payload.decode()
        
        # Expect to get dictionaray format
        try:
            data = json.loads(mesg)
        except:
            data = "Invalid json format: {}".format(mesg)

        # Do something
        if ICAP_CONF["TOPIC_REC_RPC"] in topic:
            request_idx = topic.split('/')[-1]
            self._rpc_event(request_idx, data)

        elif ICAP_CONF["TOPIC_SND_ATTR"] in topic:
            self._attr_event(data)

    def on_publish(self, client, userdata, result):
        pass

    # Send Data Usage

    def send_data(self, data: dict, topic: str):
        try:
            self.client.publish(topic, json.dumps(data))
        except Exception as e:
            log.exception(e)

    def send_attr(self, data: dict, topic:str=ICAP_CONF["TOPIC_SND_ATTR"]):
        try:
            self.client.publish(topic, json.dumps(data), retain=False)
        except Exception as e:
            log.exception(e)

    def send_tele(self, data: dict, topic:str=ICAP_CONF["TOPIC_SND_TEL"]):
        try:
            # QoS level 1 guarantees that a message is delivered at least one time to the receiver
            self.client.publish(topic, json.dumps(data), qos=1, retain=False)
            time.sleep(0.1)

        except Exception as e:
            log.exception(e)

# --------------------------------------------------------
# DEPLOYER

class ICAP_DEPLOYER(MODEL_URL_DEPLOYER):
    """ Deployer for iCAP Model """

    def __init__(self, data:dict) -> None:
        super().__init__(data["sw_url"])

        # Get Basic Data
        self.title          = data["sw_title"]
        self.ver            = data["sw_version"]
        self.tag            = data["sw_tag"]
        self.url            = data["sw_url"]

        # NOTE: checksum will lose sometime.
        self.checksum       = data["sw_checksum"]
        self.checksum_type  = data["sw_checksum_algorithm"]
        self.descr          = data["sw_description"]    
        self.package_id     = data.get("sw_package_id", "None")    
        
        # Get Description Data
        self.platform       = self.descr.get("applyProductionModel")
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
        SERV_CONF["PROC"][self.uid]["name"] = os.path.basename(self.file_folder)

    def finished_event(self):
        super().finished_event()
        self.is_finish = True

    def push_mesg(self):
        """Not only push websocket but also push to mqtt for iCAP ( Thingboard ) """

        # =============================
        # WebSocket
        super().push_mesg()
        
        # =============================
        # MQTT

        # Check status
        stats = SERV_CONF['PROC'][self.uid]['status']
        
        # Check message
        run_mesg, err_mesg = SERV_CONF['PROC'][self.uid]['message'], ""
        if stats == self.S_FAIL:
            err_mesg = run_mesg
            run_mesg = ""

        # Conbine data
        icap_data = {
            "sw_state": stats,
            "sw_error": err_mesg,
            "sw_message": run_mesg,     # Not a standard key
            "sw_package_id": self.package_id }
        
        # Finish
        if self.is_finish:
            icap_data.update({
                "current_sw_title": self.title,
                "current_sw_version": self.ver })
        
        
        SERV_CONF["ICAP"].send_tele(icap_data)

# --------------------------------------------------------
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

    if isinstance(resp_data, str):
        log.warning('Convert string response to json with key `data`')
        resp_data = { KEY_RESP_DATA: resp_data }
    
    # Merge data  
    data.update(resp_data)
    return response_status(code), data


def get_icap_task_info() -> List[dict]:
    """Get (Convert) iCAP Task information

    Returns:
        List[dict]: task info for icap
    """
    orig_task_info = task_handler.get_task_info()

    # icap_task_info = []
    # for items in orig_task_info:
    #     icap_task_info.append({
    #     "name": items["task_name"],
    #     "uuid": items["task_uid"],
    #     "status": items["status"],
    #     "model": items["model_name"],
    #     "tag": items["model_type"],
    #     "error": {},
    #     "application": items["app_name"]
    # })

    return orig_task_info

# --------------------------------------------------------
# Send Request Function

def send_post_api(trg_url, data, h_type='json', timeout=5, stderr=True):
    """ Using request to simulate POST method """
    
    try:
        resp = requests.post(trg_url, data=json.dumps(data), headers=HEAD_MAP[h_type], timeout=timeout)
        return resp_to_json(resp)

    except Exception as e:
        return request_exception(exception=e, calling_api=trg_url) 


def send_get_api(trg_url, h_type='json', timeout=5):
    """ Using request to simulate GET method """

    try:
        resp = requests.get(trg_url, headers=HEAD_MAP[h_type], timeout=timeout)
        return resp_to_json(resp)
    
    except Exception as e:
        return request_exception(exception=e, calling_api=trg_url) 

# --------------------------------------------------------
# Register Thingsboard Function

def update_config_for_icap( 
        tb_url: str, 
        tb_port: str, 
        device_name: str,
        config_path: str = SERV_CONF["CONFIG_PATH"]
    ) -> Union[dict, None]:
    
    # Read config content
    with open(config_path, "r") as f:
        config = json.load(f)

    # Update config 
    config["ICAP"]["DEVICE_NAME"] = device_name
    config["ICAP"]["HOST"] = tb_url
    config["ICAP"]["PORT"] = tb_port
    
    # Modify and save
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    log.warning('Updated ivit-i.json for iCAP')
    log.warning(f'\t- URL:PORT -> {tb_url}:{tb_port}')
    log.warning(f'\t- DEVICE_NAME: {device_name}')


def register_tb_device(
    tb_url: str, 
    tb_port: str,
    device_name: str,
    platform: str = SERV_CONF["PLATFORM"],
    api_route: str = ICAP_CONF["API_REG_DEVICE"]
    ):
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

    # Get Register URL
    register_url = f"{tb_url}:{tb_port}{api_route}"
    log.debug(f'Get thingboard url: {register_url}')
    
    # Check platform
    if platform == "jetson":
        platform = "nvidia" # Jetson have to mapping to nvidia
    
    # Concat device name
    device_type = "iVIT-I"
    if device_name == "":
        device_name = gen_uid()

    device_name = device_name \
        if device_type in device_name \
            else f"{device_type}-{device_name}" 
      
    # Get send data
    send_data = { 
        TB_KEY_NAME  : device_name,
        TB_KEY_TYPE  : device_type,
        TB_KEY_ALIAS : device_name,
        TB_KEY_MODEL : platform 
    }
    
    # Make sure http is in the header
    header = "http://"
    if ( not header in register_url ): register_url = header + register_url

    # Register via HTTP
    timeout = 5
    dict_printer(title='Registering iCAP with data', data=send_data)
    ret, data = send_post_api(register_url, send_data, timeout=timeout, stderr=False)


    # Register failed
    if not ret:
        dict_printer(
            title='Registering iCAP ... Failed !',
            data={  "URL": register_url,
                    "TOKEN": TB_KEY_TOKEN,
                    "Response": data },
            content_maximum=500,
            level='warn')
        return False, data
    
    # Register success
    data = data["data"]
    receive_data = {
        "STATUS": True,
        "CREATE_TIME": data[TB_KEY_TIME],
        "DEVICE_ID": data[TB_KEY_ID],
        "ACCESS_TOKEN": data[TB_KEY_TOKEN],
        "DEVICE_TYPE": device_type
    }

    # Update to ICAP_CONF
    ICAP_CONF.update(send_data)
    ICAP_CONF.update(receive_data)
    ICAP_CONF["HOST"] = tb_url
    ICAP_CONF["PORT"] = tb_port
    ICAP_CONF["DEVICE_NAME"] = device_name
    
    # Update config
    update_config_for_icap(
        tb_url=tb_url,
        tb_port=tb_port,
        device_name=device_name
    )

    dict_printer(   title="Registering iCAP ... Pass !",
                    data=receive_data)

    return True, data


def init_icap(
    tb_url: str, 
    tb_port: str, 
    device_name: str) -> tuple:
    """ Initialize iCAP """
    
    if tb_url=="":
        log.warning("Not setup iCAP url ...")
        return None

    # Try to Register
    flag, data = register_tb_device(
        tb_url=tb_url,
        tb_port=tb_port,
        device_name=device_name
    )
    
    # Not success
    icap_handler = None
    if not flag:
        SERV_CONF["ICAP"] = icap_handler
        log.warning('Init iCAP Failed! Set SERV_CONF[ICAP] to None')
        return flag, data

    # If success
    icap_handler = ICAP_HANDLER( 
        host=ICAP_CONF["HOST"], 
        port=ICAP_CONF["MQTT_PORT"],
        token=ICAP_CONF["ACCESS_TOKEN"] )

    SERV_CONF["ICAP"] = icap_handler
    # SERV_CONF.update({"ICAP":icap_handler})
    SERV_CONF["ICAP"].start()
    log.info("Update ICAP Object into {}".format(SERV_CONF.get_name))
    return flag, data


# --------------------------------------------------------
# Send Basic Attribute

def send_basic_attr():
    try:
        if ('ICAP' in SERV_CONF) and not (SERV_CONF['ICAP'] is None):
            
            # SERV_CONF['ICAP'].send_attr(data={
            #     'ivitTask': task_handler.get_task_info()
            # })
            send_data = {
                'ivitTask': get_icap_task_info(),
                'ivitUrl': f'{get_address()}:{SERV_CONF["WEB_PORT"]}'
            }
            SERV_CONF['ICAP'].send_attr(data=send_data)
        else:
            pass
            # log.warning('MQTT not setup ...')
    except Exception as e:
        log.warning(handle_exception(e))


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