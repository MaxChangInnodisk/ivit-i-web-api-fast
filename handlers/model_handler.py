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
import subprocess as sp

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

        # Model Type
        self.model_type = ""
        
        # Convert Process
        self.process = None

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
        """ Push message

        - Workflow
            1. Print Status
            2. If WebSocket exists then push message via `WS_CONF["WS"].send_json()`
        """

        print(' '*80, end='\r') # Clear console
        print(SERV_CONF["PROC"][self.uid]['status'], end='\r')

        if WS_CONF.get("WS") is None: return
        try:
            asyncio.run( WS_CONF["WS"].send_json( 
                ws_msg(type="PROC", content=SERV_CONF["PROC"])) )
        except Exception as e:
            log.exception(e)

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
        if not ( self.platform in [ 'nvidia', 'jetson' ] or SERV_CONF['FRAMEWORK'] == 'tensorrt'):
            return
        
        # Parse data
        info = parse_model_folder(self.file_folder)
        model_type = info.get("type")
        json_path  = info.get('json_path', "")
        model_name = get_onnx_and_darknet_path(info.get('meta_data'))
        model_path = os.path.join(self.file_folder, model_name)

        # Check data
        if model_path == "":
            raise FileNotFoundError("Could not find model.")
        elif json_path == "":
            raise FileNotFoundError("Could not find configuration.")

        # Define keywords
        keywords = {
            "CLS": {
                "&&&& RUNNING": 10,
                "Finish parsing network model": 20,
                "Total Activation Memory": 50,
                "Engine built in": 80,
                "Starting inference": 90,
                "PASSED TensorRT.trtexec": 100 },
            "OBJ": {
                "Darknet plugins are ready": 10,
                "Saving ONNX file...": 20,
                "Building ONNX graph...": 30,
                "Saving ONNX file": 40,
                "Building the TensorRT engine": 80,
                "ONNX 2 TensorRT ... Done": 100, } }

        # Start convert
        process = converting_process( model_path, model_type )

        # Capture process
        for line in iter(process.stdout.readline, b''): 
            
            # Finish
            if process.poll() != None: break

            # Decode and record
            line = line.rstrip().decode()
            line = ' '.join(line.split(' ')[2:])
            log.debug(line)

            # Check keyword and percentage
            for keyword, percentage in keywords[model_type].items():
                
                if not (keyword in line): continue

                current_status = f"{self.S_CONV} ( {percentage}% )"
                self.update_status(
                    status=current_status,
                    message=line)

        self.t_convert = time.time() - t_convert_start

    def parse_event(self):
        """Parsing Event
        
        - Workflow
            1. Extract ZIP.
            2. Remove ZIP.
        """

        # Extract
        with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
            zip_ref.extractall(self.file_folder)
        
        # Remove Zip File
        os.remove(self.file_path)

    def finished_event(self):
        """Finish Event 
        
        - Workflow
            1. Parse model folder again.
            2. Add model information into database
            3. Update model data into `SERV_CONF["PROC"][{uid}]["data"]` 
        """
        
        model_info = parse_model_folder(model_dir=self.file_folder)
        model_data = add_model_into_db(model_info=model_info)

        SERV_CONF["PROC"][self.uid].update({
            "data":model_data })
              
    def deploy_event(self):
        try:
            t_down = time.time()
            log.info('Downloading')
            self.update_status(self.S_DOWN)
            self.download_event()
            self.performance['download'] = time.time() - t_down

            t_parse = time.time()
            log.info('Parsing')
            self.update_status(self.S_PARS)
            self.parse_event()
            self.performance['parse'] = time.time() - t_parse

            t_convert = time.time()
            log.info('Converting')
            self.update_status(self.S_CONV)
            self.convert_event()
            self.performance['convert'] = time.time() - t_convert

            self.finished_event()
            self.update_status(self.S_FINISH)  
            log.info('Finished !!!')

        except Exception as e:
            log.exception(e)
            self.update_status(status=self.S_FAIL, message=handle_exception(e))

    def _create_thread(self):
        """ Create a thread which will run self.deploy_event at once """
        self.deploy_thread = threading.Thread(target=self.deploy_event, daemon=True)
        log.warning('Created deploy thread')

    def start_deploy(self):
        self.deploy_thread.start()


class MODEL_ZIP_DEPLOYER(ModelDeployerWrapper):
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


class MODEL_URL_DEPLOYER(ModelDeployerWrapper):
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

def parse_model_data(data: dict) -> dict:
    """Parse model data from database

    Args:
        data (dict): data from database

    Returns:
        dict: the model data
    """
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
    
    ret = [ parse_model_data(model) for model in data ]
    return ret


def get_model_type_from_arch(arch):
    """ Get type ( [ CLS, OBJ, SEG ] ) from training configuration which provided by iVIT-T """

    if "yolo" in arch:
        return MODEL_CONF["OBJ"]

    elif "resnet" in arch or "vgg" in arch or "mobile" in arch:
        return MODEL_CONF["CLS"]


def get_mdoel_type_from_json(json_path:str)-> str:
    """Get model type from json file

    Args:
        json_path (str): path to json configuration

    Returns:
        str: model type
    """
    with open(json_path, newline='') as jsonfile:
        train_config = json.load(jsonfile)
        arch = train_config['model_config']['arch']
        model_type = get_model_type_from_arch(arch) # CLS, OBJ
    return model_type


def parse_model_folder(model_dir):
    """ Parsing Model folder which extracted from ZIP File
    
    Samples
        * Notice: `.onnx` and `.weights` will store at meta_data.

            ```json
            {
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
            ```
    """

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
    model_exts = [ 
        MODEL_CONF["TRT_MODEL_EXT"], 
        MODEL_CONF["IR_MODEL_EXT"], 
        MODEL_CONF["XLNX_MODEL_EXT"], 
        MODEL_CONF["HAI_MODEL_EXT"] ]
    framework = [ 
        MODEL_CONF["NV"], 
        MODEL_CONF["INTEL"], 
        MODEL_CONF["XLNX"], 
        MODEL_CONF["HAILO"] ]
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
                
                if SERV_CONF['PLATFORM'] == 'intel':
                    ret['arch'] = 'yolo' if 'v3' in arch else arch
                else:
                    ret['arch'] = arch

                ret['type'] = get_model_type_from_arch( ret['arch'] )  
                
                # Basic Parameters
                ret['input_size'] = train_config['model_config']["input_shape"]
                ret['preprocess'] = train_config['train_config']['datagenerator'].get("preprocess_mode")
                
                # Parsing Anchors
                anchors = train_config.get('anchors', None)
                if anchors is None:
                    ret['anchors'] = anchors
                else:
                    ret['anchors'] = [ int(float(val.strip())) \
                        for val in anchors.strip().split(',')
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

    if ret['json_path'] == "":
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
    log.info('Initialize Database\'s Models')
    ts = time.time()

    # Get Model Folder and All Model
    model_root = os.path.realpath( model_dir )
    model_dirs = [ os.path.join(model_root, model) \
                    for model in os.listdir( model_root )]

    # Parse all file 
    for model_dir in model_dirs:
        # Ensure it's folder
        if not os.path.isdir(model_dir): continue
        # Get model information
        model_info = parse_model_folder(model_dir)
        # Try to add into database
        try: 
            add_model_into_db(model_info=model_info)
            log.debug(f"Add model into database ({model_info['name']})")
        except FileExistsError:
            log.debug(f"Model already in database ({model_info['name']})")

    # End
    te = time.time()
    log.info(f'Initialize Database\'s Models ... Done ({round(te-ts, 3)}s)')


def add_model_into_db(model_info:dict, model_uid: str=None, db_path:str=SERV_CONF["DB_PATH"]):
    """ Add One Model Information Into Database """
    
    # Get uuid
    model_uid = model_uid if model_uid else gen_uid(model_info["name"])

    # Check uuid is exist or not
    data = select_data(table="model", data="uid", condition=f"WHERE uid='{model_uid}'")
    if len(data)!=0:
        raise FileExistsError(f"Model is exist in database. ({model_info['name']}: {model_uid})")

    # Get classes
    with open(model_info["label_path"], 'r') as f:
        classes = [ re.sub('[\'"]', '', line.strip()) for line in f.readlines() ]
        classes = re.sub('["]', '\'', json.dumps(classes))

    # Prepare data
    model_db_data = {
        "uid": model_uid,
        "name": model_info["name"],
        "type": model_info["type"],
        "model_path": model_info["model_path"],
        "label_path": model_info["label_path"],
        "json_path": model_info["json_path"],
        "classes": classes,
        "input_size": model_info["input_size"],
        "preprocess": model_info["preprocess"],
        "default_model": model_info.get("default_model", 0),
        "meta_data": model_info }

    # Add into database
    insert_data(
        table="model",
        data=model_db_data,
        replace=True )

    # Return 
    return model_db_data


def add_model(file:File, url:str):
    """ Add AI Model With AddModelHelper """
    
    if (url == "") or (url is None):
        deployer = MODEL_ZIP_DEPLOYER(file=file)
    else:
        deployer = MODEL_URL_DEPLOYER(url=url)

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


def converting_process(model_path:str, model_type:str) -> sp.Popen:
    """Do convert process

    Args:
        model_path (str): path to model, should be [.onnx, .weight]
        model_type (str): the model type

    Returns:
        sp.Popen: the process object
    """
    
    # Get pure model name 
    model_name, model_ext = os.path.splitext(model_path)

    # Combine tensorrt engine path
    trg_model_path = model_name + '.trt'

    # Get correct command line
    if model_type == "CLS":
        # Classification
        cmd = [ 
            "/usr/src/tensorrt/bin/trtexec", 
            f"--onnx={model_path}", 
            f"--saveEngine={os.path.realpath(trg_model_path)}" ]
    else:
        cmd = [ "yolo-converter", f"{model_name}" ]

    process_id = sp.Popen(args=cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    log.warning(f"Run Convert Process ( { process_id } ): {' '.join(cmd)}")
    return process_id

def get_onnx_and_darknet_path(meta_data:list) -> str:
    """Help to find the onnx or darkent file.

    Args:
        meta_data (list): meta_data is get from `parse_model_folder` function.

    Raises:
        FileNotFoundError: if not find target model then raise error.

    Returns:
        str: file name ( relatived path )
    """
    for file_name in meta_data:
        ext = os.path.splitext(file_name)[1]
        if ext in [ '.onnx', '.weights' ]:
            return file_name
    raise FileNotFoundError("Could not find onnx or weight file.")

def convert_model(model_folder_path:str, background=False) -> sp.Popen:
    """A workflow for Cconvert TensorRT model 

    Args:
        model_folder_path (str): path to model folder.

    Workflow:
        1. Check the framework is tensorrt or not
        2. Parsing each file in `model_folder_path`
            * Model File ( .onnx, .weights )
            * Json File ( .json )
        3. Get model type from json file.
        4. Start converting.
        5. If run in background then return process, if not then wait for it.

    Returns:
        sp.Popen: the process object
    """

    # Double check
    if SERV_CONF['FRAMEWORK'] != 'tensorrt':
        log.warning('No need to convert ...')
        return

    # Parse data
    info = parse_model_folder(model_folder_path)

    model_type = info.get("type")
    json_path  = info.get('json_path', "")
    
    model_name = get_onnx_and_darknet_path(info.get('meta_data'))
    model_path = os.path.join(model_folder_path, model_name)

    # Check data
    if model_path == "":
        raise RuntimeError("Could not find model.")
    elif json_path == "":
        raise RuntimeError("Could not find configuration.")

    # Start convert
    process = converting_process(
        model_path=model_path,
        model_type=model_type
    )

    # If not runing in background
    if not background:
        log.warning('Waitting for converting ...')
        for line in iter(process.stdout.readline, b''): 
            if process.poll() != None: break
            print(line.rstrip().decode())
            
        log.info('Converted model !')
        
    return process
