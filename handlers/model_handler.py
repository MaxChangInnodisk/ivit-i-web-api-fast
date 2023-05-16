# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

""" Model Handler
1. parsing AI model from folder
2. add information into database
"""

import sys, os, time, json, re, shutil, zipfile, abc, wget
import asyncio, threading
import requests

import logging as log
from fastapi import File

try:
    from ..common import SERV_CONF, MODEL_CONF, WS_CONF
    from ..utils import gen_uid, load_db_json
except:
    from common import SERV_CONF, MODEL_CONF, WS_CONF
    from utils import gen_uid, load_db_json

from .db_handler import select_data, insert_data, update_data, delete_data
from .mesg_handler import handle_exception, simple_exception, ws_msg

# Model Deployer

class ModelDeployerWrapper(abc.ABC):
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

    def __init__(self) -> None:
        """
        uid:
        platfrom:
        status:
        benchmark:
        """
        super().__init__()
        
        # Basic Parameters
        self.uid = gen_uid()
        self.platform = SERV_CONF.get("PLATFORM")
        assert not (self.platform is None), "Make sure \
            SERV_CONF is already update by init_config() function\
            Which will update key and value from ivit-i.json" 
        
        # Status Parameters
        self.status = self.S_INIT

        # File Parameters
        self.file_name = ""
        self.file_path = ""
        self.model_name = self.file_folder = ""

        # Benchmark
        self.performance = {
            "download": None,
            "parse": None,
            "convert": None,
        }

        # Initialize
        self._update_uid_into_share_env()

        # Thread Placeholder and Create Thread
        self.deploy_thread = None
        self._create_thread()

    def _update_uid_into_share_env(self):
        """ Updae Environment Object """
        if SERV_CONF.get("PROC") is None:
            SERV_CONF.update({"PROC": {}})
        if SERV_CONF["PROC"].get(self.uid) is None:
            SERV_CONF["PROC"].update({
                self.uid: { 
                    "status": self.status,
                    "name": self.model_name
                }
            })
    
    def push_mesg(self):
        """ Push message up to front end, 
        default is websocket.
        You can modify by yourself 
        """

        if WS_CONF.get("WS") is None:
            print(SERV_CONF["PROC"])
            return
        
        asyncio.run( WS_CONF["WS"].send_json( ws_msg(type="PROC", content=SERV_CONF["PROC"])) )

    def update_status(self, status:str, message: str="", push_mesg:bool=True):
        """ Update Status and push message to front end """

        self.status = status
        
        SERV_CONF["PROC"][self.uid].update({
            "status": status,
            "message": message,
            "performace": self.performance
        })

        if push_mesg:
            self.push_mesg()

    def download_event(self):
        pass

    def convert_event(self):
        """ Convert Model Event """
        # Update SERV_CONF
        self.update_status(self.S_CONV)

        t_convert_start = time.time()

        # Check Platform
        if not ( self.platform in [ 'nvidia', 'intel' ]):
            return
        
        # Do Convert

        self.t_convert = time.time() - t_convert_start

    def parse_event(self):
        """ Parse ZIP File """

        # Extract
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            zip_ref.extractall(self.file_folder)
        
        # Remove Zip File
        os.remove(self.file_path)

    def finished_event(self):
        """ Finish Event """
        
        model_info = parse_model_folder(model_dir=self.file_folder)
        model_data = add_model_into_db(models_information={
            model_info['name']:model_info
        })
        SERV_CONF["PROC"][self.uid].update({
            "data":model_data })
              
    def deploy_event(self):
        try:
            t_down = time.time()
            self.update_status(self.S_DOWN)
            self.download_event()
            self.performance['download'] = time.time() - t_down

            t_parse = time.time()
            self.update_status(self.S_PARS)
            self.parse_event()
            self.performance['parse'] = time.time() - t_parse

            t_convert = time.time()
            self.update_status(self.S_CONV)
            self.convert_event()
            self.performance['convert'] = time.time() - t_convert

            self.finished_event()
            self.update_status(self.S_FINISH)  

        except Exception as e:
            log.exception(e)
            self.update_status(status=self.S_FAIL, message=handle_exception(e))

    def _create_thread(self):
        """ Create a thread which will run self.deploy_event at once """
        self.deploy_thread = threading.Thread(target=self.deploy_event, daemon=True)
        log.warning('Created deploy thread')

    def start_deploy(self):
        self.deploy_thread.start()


class ZIP_DEPLOYER(ModelDeployerWrapper):
    """ Deployer for ZIP Model """
    def __init__(self, file:File) -> None:
        super().__init__()
        
        self.file = file
        self.file_name = self.file.filename                   
        self.file_path = os.path.join( 
            SERV_CONF["MODEL_DIR"], self.file_name )
        self.file_folder = os.path.splitext(self.file_path)[0]

    def download_event(self):
        """ Download file via FastAPI """

        with open(self.file_name, "wb") as buffer:
            shutil.copyfileobj(self.file.file, buffer)
        shutil.move(self.file_name, self.file_path)
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError('Save ZIP File Failed')

        SERV_CONF["PROC"][self.uid]["name"] = os.path.basename(self.file_folder)


class URL_DEPLOYER(ModelDeployerWrapper):
    """ Deployer for URL Model """
    def __init__(self, url:str) -> None:
        super().__init__()

        self.url = url
        
        # Update Download Parameters
        self.tmp_proc_rate = 0  # avoid keeping send the same proc_rate
        self.push_rate = 10
        self.push_buf = None

    def _download_progress_event(self, current, total, width=80):
        proc_rate = int(current / total * 100)
        proc_mesg = f"{self.S_DOWN} ( {proc_rate}% )"

        if ((proc_rate%self.push_rate)==0 and proc_rate!=self.tmp_proc_rate) :
            self.tmp_proc_rate = proc_rate
            self.update_status(status=proc_mesg)

    def download_event(self):
        """ Download file via URL from iVIT-T """
        self.update_status(self.S_DOWN)
        
        self.file_name = wget.download( self.url, bar=self._download_progress_event)
        self.file_path = os.path.join( SERV_CONF["MODEL_DIR"], self.file_name)
        shutil.move( self.file_name, self.file_path )
        self.file_folder =  os.path.splitext( self.file_path )[0]
        
        SERV_CONF["PROC"][self.uid]["name"] = os.path.basename(self.file_folder)



# Helper Function

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
        "meta_data": json.loads(data[9]),
        "annotation": data[10]
    }


def get_model_info(uid: str=None):
    """ Get Model Information from database """
    if uid == None:    
        data = select_data(table='model', data="*")
    else:
        uid = str(uid.strip())
        data = select_data(table='model', data="*", condition=f"WHERE uid='{uid}'")
    
    print(uid, data)
    ret = [ parse_model_data(model) for model in data ]
    return ret


def get_model_tag_from_arch(arch):
    """ Get type ( [ CLS, OBJ, SEG ] ) from training configuration which provided by iVIT-T """

    if "yolo" in arch:
        return MODEL_CONF["OBJ"]

    elif "resnet" in arch or "vgg" in arch or "mobile" in arch:
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
        log.error("[{}] Can not find JSON Configuration".format(model_dir))

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
        model_uid = gen_uid(model_name)
        with open(model_info['label_path'], 'r') as f:
            classes = [ re.sub('[\'"]', '', line.strip()) for line in f.readlines() ]
            classes = re.sub('["]', '\'', json.dumps(classes))
        
        model_db_data = {
                "uid": model_uid,
                "name": model_info['name'],
                "type": model_info['type'],
                "model_path": model_info['model_path'],
                "label_path": model_info['label_path'],
                "json_path": model_info['json_path'],
                "classes": classes,
                "input_size": model_info['input_size'],
                "preprocess": model_info['preprocess'],
                "meta_data": model_info,
        }

        insert_data(
            table='model',
            data=model_db_data,
            replace=True)
    
    return model_db_data

def add_model(file:File, url:str):
    """ Add AI Model With AddModelHelper """
    
    if (url == "") or (url is None):
        deployer = ZIP_DEPLOYER(file=file)
    else:
        deployer = URL_DEPLOYER(url=url)

    deployer.start_deploy()

    while(deployer.status != deployer.S_PARS):
        print(f"{deployer.status:20}", end='\r')

    return {
        "uid": deployer.uid,
        "status": deployer.status
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

    