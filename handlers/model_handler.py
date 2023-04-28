# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" Model Handler
1. parsing AI model from folder
2. add information into database
"""

import sys, os, time, json, re, shutil, zipfile, wget
import asyncio, threading
import requests

import logging as log
from fastapi import File
from ..common import SERV_CONF, MODEL_CONF, WS_CONF
from ..utils import gen_uuid, load_db_json

from .db_handler import select_data, insert_data, update_data, delete_data
from .mesg_handler import handle_exception, simple_exception


class AddModelHelper(object):
    """ Model Helper 
    1. Download ZIP
    2. Extract ZIP
    3. Convert Model if need
    4. Initailize Model into Database
    """
    S_FAIL = "Failure"
    S_INIT = "Waiting"
    S_DOWN = "Downloading"      # Downloading ({N}%)
    S_PARS = "Verifying"
    S_CONV = "Converting"       # Converting ({N}%)
    S_FINISH = "Success"

    def __init__(self, uid:str=None, file:File=None, url:str=None, ws_mode:bool=True, mqtt_mode:bool=False) -> None:
        
        # DoubleCheck
        assert not (file is None) or not(url is None), "Make sure the file or url is available." 

        # Basic Parameters
        self.uid = uid if uid else gen_uuid()
        self.file = file
        self.url = url
        self.platform = SERV_CONF["PLATFORM"]        
        self.status = self.S_INIT
        self.file_name = ""
        self.file_path = ""
        self.file_folder = ""

        # Update SERV_CONF information
        if SERV_CONF.get('MODEL') is None:
            SERV_CONF.update({'MODEL': {}})
        if SERV_CONF['MODEL'].get(self.uid) is None:
            SERV_CONF['MODEL'].update({
                self.uid: { 
                    "status": self.status,
                    "name": self.file_folder
                }
            })

        # Switcher
        self.ws_mode = ws_mode
        self.mqtt_mode = mqtt_mode

        # Status PlaceHolder
        self.status = ""

        # Benchmark
        self.t_download = None
        self.t_parse = None
        self.t_convert = None
        self.t_init = None
        
        # Update Download Parameters
        self.tmp_proc_rate = 0  # avoid keeping send the same proc_rate
        self.push_rate = 10
        self.push_buf = None

        # Create Thread
        self.t = threading.Thread(target=self.deploy_event, daemon=True)

    def start(self):
        self.t.start()

    def update_status(self, status:str, message: dict={}, send_data:bool=True):
        self.status = status
        SERV_CONF['MODEL'][self.uid].update({
            'status': status,
            'message': message
        })

        if send_data:
            self.send_ws()

    def send_ws(self):
        if WS_CONF.get("WS") is None:
            return
        
        asyncio.run( WS_CONF["WS"].send_json({"MODEL": SERV_CONF['MODEL']}) )

    def deploy_event(self):
        try:
            self.download_event()
            self.parse_event()
            self.convert_event()
            self.finished_event()
        except Exception as e:
            log.exception(e)
            self.update_status(status=self.S_FAIL, message=handle_exception(e))

    def __download_fastapi_file(self):
        """ Download file via FastAPI """

        self.file_name = self.file.filename            
        self.file_path = os.path.join( 
            SERV_CONF["MODEL_DIR"], self.file_name )
        self.file_folder = os.path.splitext(self.file_path)[0]

        with open(self.file_name, "wb") as buffer:
            shutil.copyfileobj(self.file.file, buffer)
        shutil.move(self.file_name, self.file_path)
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError('Save ZIP File Failed')

        SERV_CONF["MODEL"][self.uid]["name"] = os.path.basename(self.file_folder)

    def __download_progress_event(self, current, total, width=80):
        proc_rate = int(current / total * 100)
        proc_mesg = f"{self.S_DOWN} ( {proc_rate}% )"

        if ((proc_rate%self.push_rate)==0 and proc_rate!=self.tmp_proc_rate) :
            self.tmp_proc_rate = proc_rate
            self.update_status(status=proc_mesg)

    def __download_url_file(self):
        """ Download file via URL from iVIT-T """
        self.file_name = wget.download( self.url, bar=self.__download_progress_event)
        self.file_path = os.path.join( SERV_CONF["MODEL_DIR"], self.file_name)
        shutil.move( self.file_name, self.file_path )
        self.file_folder =  os.path.splitext( self.file_path )[0]
        SERV_CONF["MODEL"][self.uid]["name"] = os.path.basename(self.file_folder)

    def download_event(self):
        """ Download from URL """
        self.update_status(self.S_DOWN)
        t_down_start = time.time()
        if self.file:
            self.__download_fastapi_file()
        else:
            self.__download_url_file()
        self.t_download = time.time() - t_down_start

    def parse_event(self):
        """ Parse ZIP File """
        self.update_status(self.S_PARS)
        t_parse_start = time.time()

        # Extract
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            zip_ref.extractall(self.file_folder)
        os.remove(self.file_path)

        # URL Mode should rename folder
        if self.file is None and self.url:
            log.warning('Rename Folder ...')
            
            self.file_folder

        self.t_parse = time.time() - t_parse_start
    
    def convert_event(self):
        """ Convert Model """
        t_convert_start = time.time()

        # Check Platform
        if not ( self.platform in [ 'nvidia', 'intel' ]):
            return
        
        # Do Convert
        
        # Update SERV_CONF
        self.update_status(self.S_CONV)
        self.t_convert = time.time() - t_convert_start
    
    def finished_event(self):
        """ Initialize Model """
        t_init_start = time.time()
        init_db_model()
        model_info = parse_model_folder(model_dir=self.file_folder)
        add_model_into_db(models_information={
            model_info['name']:model_info
        })
        # Update SERV_CONF
        self.update_status(self.S_FINISH)
        self.t_init = time.time()-t_init_start

    def get_benchmark(self):
        """ Provide process time of each event """
        return {
            "download_time": self.t_download,
            "parse_time": self.t_parse,
            "convert_time": self.t_convert,
            "init_time": self.t_init
        }


def parse_model_data(data: dict):
    return {
        "uid": data[0],
        "name": data[1],
        "type": data[2],
        "model_path": data[3],
        "label_path": data[4],
        "json_path": data[5],
        "classes": data[6],
        "input_size": data[7],
        "preprocess": data[8],
        "meta_data": load_db_json(data[9]),
        "annotation": data[10]
    }


# Helper Function
def get_model_info(uid: str=None):
    """ Get Model Information from database """
    if uid == None:    
        data = select_data(table='model', data="*")
    else:
        data = select_data(table='model', data="*", condition=f"WHERE uid='{uid}'")
    
    ret = [ parse_model_data(model) for model in data ]
    return ret


def get_model_tag_from_arch(arch):
    """ Get type ( [ CLS, OBJ, SEG ] ) from training configuration which provided by iVIT-T """

    if "yolo" in arch:
        return MODEL_CONF["OBJ"]

    elif "resnet" in arch or "vgg" in arch:
        return MODEL_CONF["CLS"]
         

def parse_model_folder(model_dir):
    """ Parsing Model folder which extracted from ZIP File """

    ret = {
        'type': '',
        'name': '',
        'arch': '',
        'framework': '',
        'model_dir': model_dir,
        'model_path': '',
        'label_path': '',
        'json_path': '',
        'config_path': '',
        'meta_data': [],
        'anchors': [],
        'input_size': '',
        'preprocess': None
    }

    # Double Check
    model_exts = [ MODEL_CONF["TRT_MODEL_EXT"], MODEL_CONF["IR_MODEL_EXT"], MODEL_CONF["XLNX_MODEL_EXT"] ]
    framework = [ MODEL_CONF["NV"], MODEL_CONF["INTEL"], MODEL_CONF["XLNX"]  ]
    assert len(framework)==len(model_exts), "Code Error, Make sure the length of model_exts and framework is the same "

    # Get Name
    ret['name'] = os.path.basename(model_dir)

    # Get Directory
    model_dir = os.path.realpath(model_dir)
    ret['model_dir'] = model_dir

    # Parse another file
    for fname in os.listdir(model_dir):
                    
        fpath = os.path.join(model_dir, fname)
        name, ext = os.path.splitext(fpath)
        
        if ext in model_exts:
            # print("\t- Detected {}: {}".format("Model", fpath))
            ret['model_path']= fpath
            ret['framework'] = framework[ model_exts.index(ext) ]

        elif ext in [ MODEL_CONF["DARK_LABEL_EXT"], MODEL_CONF["CLS_LABEL_EXT"], ".names", ".txt" ]:
            # print("\t- Detected {}: {}".format("Label", fpath))
            ret['label_path']= fpath

        elif ext in [ MODEL_CONF["DARK_JSON_EXT"], MODEL_CONF["CLS_JSON_EXT"] ]:
            # print("\t- Detected {}: {}".format("JSON", fpath))
            ret['json_path']= fpath
            
            # get type
            with open(fpath, newline='') as jsonfile:
                train_config = json.load(jsonfile)
                arch = train_config['model_config']['arch']
                ret['arch'] = 'yolo' if 'v3' in arch else arch  
                ret['type'] = get_model_tag_from_arch( ret['arch'] )  
                
                # Basic Parameters
                ret['input_size'] = train_config['model_config']["input_shape"]
                ret['preprocess'] = train_config['train_config']['datagenerator'].get("preprocess")
                
                # INTEL 
                if 'anchors' in train_config:
                    ret['anchors'] = [ int(val.strip()) \
                        for val in train_config['anchors'].strip().split(',')
                    ]

        elif ext in [ MODEL_CONF["DARK_CFG_EXT"] ]:
            # print("\t- Detected {}: {}".format("Config", fpath))
            ret['config_path']= fpath

            # NVIDIA
            with open(fpath, newline='') as txtfile:
                for line in txtfile.readlines()[::-1]:
                    if not 'anchors' in line:
                        continue

                    str_anchors = line.split('=')[1].strip()
                    ret['anchors'] = \
                    [ int(a.strip()) for a in str_anchors.split(',') ]
    
        else:
            # print("\t- Detected {}: {}".format("Meta Data", fpath))
            ret['meta_data'].append(fpath)

    if "" in [ ret['json_path'] or ret['model_path'] ]:
        log.error("[{}] Can't find JSON Configuration".format(model_dir))

    return ret


def init_db_model(model_dir:str=SERV_CONF["MODEL_DIR"], add_db:bool=True) -> dict:
    """ Parse model from folder 
    - arguments
        - model_dir
            - type: string
            - desc: the directory
    - return
        - models_information
            - type: dict
            - desc: all information of each model
    """

    t_start = time.time()
    models_information = {}

    # Get Model Folder and All Model
    model_root = os.path.realpath( model_dir )
    model_dirs = [ os.path.join(model_root, model) for model in os.listdir( model_root )]

    # Parse all file 
    for model_dir in model_dirs:
        if not os.path.isdir(model_dir):
            continue

        try:    
            models_information.update({  
                os.path.basename(model_dir): parse_model_folder(model_dir) })
            
        except Exception as e:
            log.exception(e)

    # for model_name, model_info in models_information.items():
    #     print('\n--------- {} --------'.format(model_name))
    #     for key, val in model_info.items():
    #         print('{:4}- {:<10}: {}'.format('', key, val))

    log.info('Initialized AI Model. Found {} AI Models ({}s)'.format(
        len(models_information),
        round(time.time()-t_start, 5)))
    
    if add_db:
        add_model_into_db(models_information)

    return models_information


def add_model_into_db(models_information:dict, db_path:str=SERV_CONF["DB_PATH"]):
    """ Add Model Information Into Database """
    
    for model_name, model_info in models_information.items():
        model_uid = gen_uuid(model_name)
        with open(model_info['label_path'], 'r') as f:
            classes = [ re.sub('[\'"]', '', line.strip()) for line in f.readlines() ]
            classes = re.sub('["]', '\'', json.dumps(classes))
            
        insert_data(
            table='model',
            data={
                "uid": model_uid,
                "name": model_info['name'],
                "type": model_info['type'],
                "model_path": model_info['model_path'],
                "label_path": model_info['label_path'],
                "json_path": model_info['json_path'],
                "classes": classes,
                "input_size": json.dumps(model_info['input_size']),
                "preprocess": model_info['preprocess'],
                "meta_data": re.sub('["]', '\'', json.dumps(model_info)),
            },
            replace=True)
    

def add_model(file:File, url:str):
    """ Add AI Model With AddModelHelper """
    
    model_helper = AddModelHelper(
        file=file,
        url=url,
        ws_mode=True,
        mqtt_mode=False
    )
    model_helper.start()

    while(True):
            print(f"{model_helper.status:20}", end='\r')
            if model_helper.status == model_helper.S_PARS:
                break

    return {
        "uid": model_helper.uid,
        "status": model_helper.status
    }


def delete_model(uid:str):

    model_data = select_data(table='model', data=["name"], condition=f"WHERE uid='{uid}'")
    
    if model_data == []:
        raise NameError('Could not find target model: {}'.format(uid))

    model_name = model_data[0][0]
    
    shutil.rmtree(os.path.join(SERV_CONF["MODEL_DIR"], model_name))
    
    delete_data(table='model', condition=f"WHERE uid='{uid}'")


if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG)

    models_information = init_db_model( model_dir="./model" )

    add_model_into_db(models_information)

    