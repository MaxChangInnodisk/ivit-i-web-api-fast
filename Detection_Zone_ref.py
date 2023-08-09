import sys, os, cv2, logging, time
import numpy as np
import uuid
import os
import threading
import math
from typing import Any, Union, get_args
from datetime import datetime
from apps.palette import palette
from ivit_i.common.app import iAPP_OBJ
import os
import numpy as np
import time
from typing import Tuple, Callable
from collections import defaultdict

# Parameters
FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
BASE_THICK      = 1         # Setup Basic Thick Value
BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.
WIDTH_SPACE = 10
HIGHT_SPACE = 10

def denorm_area_point(frame: np.ndarray, norm_points: list) -> list:
    ret_point = []
    return ret_point

class DrawTooler:
    """ Draw Tool for Label, Bounding Box, Area ... etc. """

    def __init__(self, label_path:str, palette:dict) -> None:
            
        # palette[cat] = (0,0,255), labels = [ cat, dog, ... ]
        self.label_palette, self.labels = \
            self._get_palette_and_labels(palette=palette, label_path=label_path)

        # Initialize draw params
        self.font_size = 1
        self.font_thick = 1
        self.line_thick = 1        
        self.area_color = [ 0,0,255 ]
        self.area_opacity = 0.4          
        self.draw_params_is_ready = False

    def _get_palette_and_labels(self, palette:dict, label_path:str) -> Tuple[dict, list]:
        """update the color palette ( self.palette ) which key is label name and label list ( self.labels).

        Args:
            palette (dict): the default palette
            label_path (str): the path to label file

        Raises:
            TypeError: if the default palette with wrong type
            FileNotFoundError: if not find label file

        Returns:
            Tuple[dict, list]: palette, labels
        """
        
        # check type and path is available
        if not isinstance(palette, dict):
            raise TypeError(f"Expect palette type is dict, but get {type(palette)}.")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Can not find label file in '{label_path}'.")
        
        # params
        ret_palette = {}
        ret_labels = []

        # update custom color if need
        custom_palette = self.params["application"].get("palette", {})
        
        # update palette and label
        idx = 1
        f = open(label_path, 'r')
        for raw_label in f.readlines():

            label = raw_label.strip()
            
            # if setup custom palette
            if label in custom_palette:
                color = custom_palette[label]
            
            # use default palette 
            else:
                color = palette[str(idx)]
            
            # update palette, labels and idx
            ret_palette[label] = color
            ret_labels.append(label)
            idx += 1

        f.close()

        return ret_palette, ret_labels

    def update_draw_params(self, frame: np.ndarray):
        
        if self.draw_params_is_ready: return

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.line_thick  = BASE_THICK + round( scale )
        self.font_thick = self.line_thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )
        self.width_space = int(scale*WIDTH_SPACE) 
        self.height_space = int(scale*HIGHT_SPACE) 

        # Change Flag
        self.draw_params_is_ready = True
        print('Updated draw parameters !!!')

    def draw_area(self, frame: np.ndarray, points: list, name: str, color = None, thick = None) -> np.ndarray:
        pass
    
    def draw_bbox(self, frame: np.ndarray, left_top: list, right_bottom: list, color = None, thick = None) -> np.ndarray:
        # Draw bbox
        pass

    def draw_label(self, frame: np.ndarray, label: str, position: list) -> np.ndarray:
        # Draw label
        pass

    

class Detection_Zone(iAPP_OBJ):

    def __init__(self, params:dict, label:str, event_save_folder:str="event", palette:dict=palette):
        
        # NOTE: Step 1. Define Application Type in __init__()
        self.app_type = 'obj' 
        
        self._init_app_params()
        self._update_palette_and_labels(
            palette = palette,
            label_path = label )
        
    # ------------------------------------------------
    # Initialize Application Parameters
    def _init_app_params(self):
        """Initailize all parameters of current application """

        # palette and label
        self.palette = {}
        self.labels = []

        # for draw result and boundingbox
        self.frame_idx = 0
        self.frame_size = []
        self.font_size  = None
        self.font_thick = None
        self.thick      = None
        self.width_space = None
        self.height_space = None
        
        # for draw area
        self.areas = []

        self.area_name = {}
        self.area_opacity = None
        self.area_color = []
        self.area_pts = {}
        self.area_cnt = {}
        
        # the input area point is normalized
        self.norm_area_pts = {}

        # control draw
        self.draw_bbox=self.params['application']['draw_bbox'] if self.params['application'].__contains__('draw_bbox') else False
        self.draw_result=self.params['application']['draw_result'] if self.params['application'].__contains__('draw_result') else False
        self.draw_area=False
        self.draw_app_common_output = True

        self.depend_on = {}
        self.app_output_data = {}

    # ------------------------------------------------
    # Update parameter in __init__

    def _update_palette_and_labels(self, palette:dict, label_path:str) -> None:
        """update the color palette ( self.palette ) which key is label name and label list ( self.labels).

        Args:
            palette (dict): the default palette
            label_path (str): the path to label file

        Raises:
            TypeError: if the default palette with wrong type
            FileNotFoundError: if not find label file

        """
        
        # check type and path is available
        if not isinstance(palette, dict):
            raise TypeError(f"Expect palette type is dict, but get {type(palette)}.")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Can not find label file in '{label_path}'.")
        
        # update custom color if need
        custom_palette = self.params["application"].get("palette", {})
        
        # update palette and label
        idx = 1
        f = open(label_path, 'r')
        for raw_label in f.readlines():

            label = raw_label.strip()
            
            # if setup custom palette
            if label in custom_palette:
                color = custom_palette[label]
            
            # use default palette 
            else:
                color = palette[str(idx)]
            
            # update palette, labels and idx
            self.palette[label] = color
            self.labels.append(label)
            idx += 1

        f.close()

    def _update_area_params(self) -> None:
        """Check and update area parameters.

        Args:
        
        Workflows:

        Returns:

        """

        def _get_depend(data) -> list:
            # check has depend_on and type is correct
            _depend_on = data.get("depend_on", None)
            if not _depend_on:
                raise ValueError(f"Can not find 'depend_on' in area setting ")
            elif _depend_on==[]:
                _depend_on = self.labels
            else:
                self._check_type(_depend_on, list)
            return _depend_on

        def _get_app_output(depend_on: list) -> list:
            return [ { "label": label, "num": 0 } for label in depend_on ]

        def _get_area_point(data, default_value= [[0,0],[1,0],[1,1],[0,1]]) -> list:
            _area_point = data.get("area_point", None)
            if not _area_point:
                raise ValueError(f"Can not find 'area_point' in config.")
            elif _area_point == []:
                _area_point = default_value
            else:
                self._check_type(_area_point, list)
            return _area_point

        def _get_area_name(data, default_value= "The defalt area") -> str:
            _name = data.get("name", None)
            if not _name:
                raise ValueError(f"Can not find 'name' in config.")
            elif _name == []:
                _name = default_value
            else:
                self._check_type(_name, str)
            return _name

        areas  = self.params["application"].get("area")
        if not areas:
            raise KeyError("Con not find area key in application config.")
        if not isinstance(areas, list):
            raise TypeError("The expected type of the area setting shout be list.")

        for idx, info in enumerate(areas):
            """
            1. Name
            2. Area Point ( Normalize Version )
            3. Depend_on
            4. Generate App Output Data
            """
            if not isinstance(info, dict):
                raise TypeError("Support area information type is dict.")
            
            new_areas = {}
            new_areas["name"] = _get_area_name(info)
            new_areas["norm_area_points"] = _get_area_point(info)
            new_areas["depend_on"] = None

            self.areas.append(new_areas)
            pass
    
    def _update_event_params(self):
        pass
    
    # ------------------------------------------------
    # Update parameter in __call__

    def _update_draw_params(self, frame:np.ndarray) -> None:
        """ Update the parameters of the drawing tool, which only happend at first time. """

        # if frame_size not None means it was already init 
        if( self.frame_idx >= 1): return None

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.thick  = BASE_THICK + round( scale )
        self.font_thick = self.thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )

        self.area_color = [ 0,0,255 ]
        self.area_opacity = 0.4  

        self.width_space = int(scale*WIDTH_SPACE) 
        self.height_space = int(scale*HIGHT_SPACE) 

    # ------------------------------------------------
    # NOTE: __call__ function is requirements
    def __call__(self, frame:np.ndarray, detection:list) -> Tuple[np.ndarray, list, list]:
        """
        1. Update basic parameters
        2. Parse detection and draw
        3. Check the event is trigger or not
        4. Return data
        """

        self._update_draw_param(frame)
        pass

if __name__=='__main__':

    import logging as log
    import sys, cv2
    from argparse import ArgumentParser, SUPPRESS
    from typing import Union
    from ivit_i.io import Source, Displayer
    from ivit_i.core.models import iDetection
    from ivit_i.common import Metric

    def build_argparser():

        parser = ArgumentParser(add_help=False)
        basic_args = parser.add_argument_group('Basic options')
        basic_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
        basic_args.add_argument('-m', '--model', required=True,
                        help='Required. Path to an .xml file with a trained model '
                            'or address of model inference service if using ovms adapter.')
        basic_args.add_argument('-i', '--input', required=True,
                        help='Required. An input to process. The input must be a single image, '
                            'a folder of images, video file or camera id.')
        available_model_wrappers = [name.lower() for name in iDetection.available_wrappers()]
        basic_args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                        type=str, required=True, choices=available_model_wrappers)
        basic_args.add_argument('-d', '--device', type=str,
                        help='Optional. `Intel` support [ `CPU`, `GPU` ] \
                                `Hailo` is support [ `HAILO` ]; \
                                `Xilinx` support [ `DPU` ]; \
                                `dGPU` support [ 0, ... ] which depends on the device index of your GPUs; \
                                `Jetson` support [ 0 ].' )

        model_args = parser.add_argument_group('Model options')
        model_args.add_argument('-l', '--label', help='Optional. Labels mapping file.', default=None, type=str)
        model_args.add_argument('-t', '--confidence_threshold', default=0.6, type=float,
                                    help='Optional. Confidence threshold for detections.')
        model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                    help='Optional. A space separated list of anchors. '
                                            'By default used default anchors for model. \
                                                Only for `Intel`, `Xilinx`, `Hailo` platform.')

        io_args = parser.add_argument_group('Input/output options')
        io_args.add_argument('-n', '--name', default='ivit', 
                            help="Optional. The window name and rtsp namespace.")
        io_args.add_argument('-r', '--resolution', type=str, default=None, 
                            help="Optional. Only support usb camera. The resolution you want to get from source object.")
        io_args.add_argument('-f', '--fps', type=int, default=None,
                            help="Optional. Only support usb camera. The fps you want to setup.")
        io_args.add_argument('--no_show', action='store_true',
                            help="Optional. Don't display any stream.")

        args = parser.parse_args()
        # Parse Resoltion
        if args.resolution:
            args.resolution = tuple(map(int, args.resolution.split('x')))

        return args
    # 1. Argparse
    args = build_argparser()

    # 2. Basic Parameters
    infer_metrx = Metric()

    # 3. Init Model
    model = iDetection(
        model_path = args.model,
        label_path = args.label,
        device = args.device,
        architecture_type = args.architecture_type,
        anchors = args.anchors,
        confidence_threshold = args.confidence_threshold )

    # 4. Init Source
    src = Source( 
        input = args.input, 
        resolution = (640, 480), 
        fps = 30 )

    # 5. Init Display
    if not args.no_show:
        dpr = Displayer( cv = True )

    # 6. Setting iApp
    app_config = {
        "application": {
						"palette": {
                        "car": [
                            105,
                            125,
                            105
                        ],
                        "truck": [
                            125,
                            115,
                            105
                        ]
                    },
            "areas": [
            
                {
                    "name": "Area0",
                    "depend_on": [
                        "car",
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.583
                        ],
                        [
                            0.658,
                            0.503
                        ],
                        [
                            0.848,
                            0.712
                        ],
                        [
                            0.356,
                            0.812
                        ]
                    ],
                    "events": {
                        "uid":"cfd1f399",
                        "title": "Traffic is very heavy",
                        "logic_operator": ">",
                        "logic_value": 1,
                    }
                },
                {
                    "name": "Area1",
                    "depend_on": [
                        "car",
                    ],
                    "area_point": [
                        [
                            0.256,
                            0.383
                        ],
                        [
                            0.538,
                            0.203
                        ],
                        [
                            0.268,
                            0.512
                        ],
                        [
                            0.456,
                            0.212
                        ]
                    ],
                    "events": {
                        "uid":"cfd1f399",
                        "title": "Traffic is very heavy",
                        "logic_operator": ">",
                        "logic_value": 1,
                    }
                }
            ]
        }
    }
    app = Detection_Zone(app_config ,args.label)

    # 7. Start Inference
    try:
        while True:
            # Get frame & Do infernece
            frame = src.read()
            
            results = model.inference(frame=frame)
            frame , app_output , event_output =app(frame,results)
            
            # infer_metrx.paint_metrics(frame)
            
            # Draw FPS: default is left-top                     
            dpr.show(frame=frame)

            # Display
            if dpr.get_press_key() == ord('+'):
                model.set_thres( model.get_thres() + 0.05 )
            elif dpr.get_press_key() == ord('-'):
                model.set_thres( model.get_thres() - 0.05 )
            elif dpr.get_press_key() == ord('q'):
                break

            # Update Metrix
            infer_metrx.update()

    except KeyboardInterrupt:
        log.info('Detected Key Interrupt !')

    finally:
        model.release()
        src.release()
        dpr.release()