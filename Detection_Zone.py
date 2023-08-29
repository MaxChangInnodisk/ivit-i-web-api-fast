import os, cv2, time
import numpy as np
import uuid
import os
import math
from typing import Union
import os
import numpy as np
import time
from typing import Tuple, List
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import json
import copy

# iVIT-I 
from ivit_i.common.app import iAPP_OBJ

# App utilities
try:
    from .palette import palette
    from .utils import ( 
        timeit, get_time,
        update_palette_and_labels,
        get_logic_map,
        denorm_area_points,
        sort_area_points,
        denorm_line_points
    )
    from .drawer import DrawTool
except:
    from apps.palette import palette
    from apps.utils import ( 
        timeit, get_time,
        update_palette_and_labels,
        get_logic_map,
        denorm_area_points,
        sort_area_points,
        denorm_line_points
    )
    from apps.drawer import DrawTool

# ------------------------------------------------------------------------    

class EventHandler:

    def __init__(   self, 
                    title: str, 
                    folder: str,
                    drawer: DrawTool, 
                    operator: str, 
                    threshold: float, 
                    cooldown: float ):
        # Basic
        self.title = title
        self.drawer = drawer
        self.threshold = threshold
        self.cooldown = cooldown

        # Logic
        self.logic_map = get_logic_map()
        self.logic_event = self.logic_map[operator]
    
        # Timer
        self.trigger_time = time.time_ns()

        # Generate uuid
        self.folder = self.check_folder(folder)
        self.uuid = str(uuid.uuid4())[:8]
        self.uuid_folder = self.check_folder(os.path.join(self.folder, self.uuid))
        
        # Dynamic Variable
        self.current_time = time.time_ns()
        self.current_value = None

        # Custom Threading Pool
        self.exec_pool = []
        self.pools = ThreadPool(processes=4)

        # Flag
        self.event_status = False

    def check_folder(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):        
            os.makedirs(folder_path)
        return folder_path

    def is_trigger(self) -> bool:
        return self.logic_event(self.current_value , self.threshold)

    def is_ready(self) -> bool:
        if (self.current_time - self.trigger_time) <= self.cooldown:
            print('Not ready, still have {} seconds'.format(round(self.current_time - self.trigger_time, 5)))
            return False
        self.trigger_time = self.current_time
        return True

    def get_pure_dets(self, detections: list):
        # return [{
        #     "xmin": det.xmin,
        #     "ymin": det.ymin,
        #     "xmax": det.xmax,
        #     "ymax": det.ymax,
        #     "label": det.label,
        #     "score": det.score
        # } for det in detections ]
        return detections

    def get_pure_area(self, area:dict) -> dict:
        return {
            'name':area['name'],
            'area_point': area['norm_area_point'],
            'output': area['output']
        }

    def save_image(self, save_path: str, image: np.ndarray) -> None:
        cv2.imwrite(save_path, image)

    def get_meta_data(self, pure_dets: list, pure_area: dict) -> dict:
        return {
            "detections": pure_dets,
            "area": pure_area
        }

    def save_meta_data(self, path: str, meta_data: dict) -> None:
        copy_data = copy.deepcopy(meta_data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(copy_data, f, ensure_ascii=False, indent=4)

    def event_behaviour(self, original: np.ndarray, meta_data: dict) -> dict:
        # timestamp is folder name
        timestamp = str(self.current_time)
        
        target_folder = self.check_folder(self.uuid_folder)
        
        # Save Data: Image, Metadata
        image_path = os.path.join(target_folder, f"{timestamp}.jpg")
        data_path = os.path.join(target_folder, f"{timestamp}.json")

        self.pools.apply_async( self.save_image, args=(image_path, original))
        self.pools.apply_async( self.save_meta_data, args=(data_path, meta_data))
        
        if self.event_status:
            self.start_time = self.current_time
            self.end_time = None
        else:
            self.end_time = self.current_time

        return {
            "uid": self.uuid,
            "title": self.title,
            "timestamp": self.current_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "event_status": self.event_status,
            "meta": meta_data,
        }

    def _reset_timestamp(self):
        self.start_time = None
        self.end_time = None

    def _print_title(self, text: str):
        print(f'\n * [{self.title.upper()}] {text}')

    def __call__(self, original: np.ndarray, value: int, detections: list, area: dict) -> Union[None, dict]:
        
        # Update variable
        self.current_time = time.time_ns()
        self.current_value = value

        # Cooldown time: avoid trigger too fast
        if not self.is_ready(): 
            self._print_title('Not ready')
            return
        
        # Event triggered
        event_is_trigger = self.is_trigger()
        if event_is_trigger and self.event_status: 
            # Event is on going then return
            return

        elif not event_is_trigger and not self.event_status:              
            # Event not triggered and event not started
            self._reset_timestamp()
            return

        # Update Status
        self.event_status = event_is_trigger
                
        self._print_title('Event {} ( Total: {})'.format(
            'Start' if self.event_status else 'Stop', value))

        # get data
        pure_dets = self.get_pure_dets(detections)
        pure_area = self.get_pure_area(area)
        meta_data = self.get_meta_data(pure_dets=pure_dets, pure_area=pure_area)

        # Event
        return self.event_behaviour(
                original= original,
                meta_data= meta_data
        )

    def __del__(self):
        # Close threading pool iterminate
        self.pools.terminate()

# ------------------------------------------------------------------------    

class Detection_Zone(iAPP_OBJ):

    def __init__(self, params:dict, label:str, event_behavi_save_folder:str="event", palette:dict=palette):
        """_summary_

        Args:
            params (dict): _description_
            label (str): _description_
            event_save_folder (str, optional): _description_. Defaults to "event".
            palette (dict, optional): _description_. Defaults to palette.
        
        Workflows:
            1. Setup app_type.
            2. Initailize all parameters.
            3. Update palette and labels.
            4. Update each area informations into self.areas.
                * name
                * depend_on
                * norm_area_point
                * output
        """
        # NOTE: Step 1. Define Application Type in __init__()
        self.app_type = 'obj' 
        
        # Initailize app params
        self.areas = []

        # Update setting
        self.params = params
        self.app_setting = self._get_app_setting(params)
        self.draw_area = self.app_setting.get("draw_area", False) 
        self.draw_bbox = self.app_setting.get("draw_bbox", True)
        self.draw_label = self.app_setting.get("draw_label", True)
        self.draw_result = self.app_setting.get("draw_result", True)
        
        # Event
        self.force_close_event = False

        # Palette and labels
        self.custom_palette = self.app_setting.get("palette", {})
        self.palette, self.labels = update_palette_and_labels(
            custom_palette = self.custom_palette,
            default_palette = palette,
            label_path = label )
 
        # init draw tool that share the same palette and labels
        self.drawer = DrawTool(
            labels = self.labels,
            palette = self.palette
        )
       
        # NOTE: main parameter object
        self.areas = self.get_areas_setting(
            app_setting= self.app_setting,
            labels= self.labels
        )
        
        # helper
        self.frame_h, self.frame_w = None, None

    # ------------------------------------------------
    # Update parameter in __init__

    def _get_app_setting(self, input_params:dict) -> dict:
        """Check the input parameters is available.

        Args:
            input_params (dict): the input parameters.

        Raises:
            TypeError: the input parameter should be dict.
            KeyError: the input parameter must with key 'application'.
            TypeError: the application content should be dict ( input_params['application'] ).

        Returns:
            dict: the data of application.
        """


        #check config type
        if not isinstance( input_params, dict ):
            raise TypeError(f"The application parameter should be dict, but get {type(input_params)}")
        
        #check config key ( application ) is exist or not and the type is correct or not
        data =  input_params.get("application", None)
        if not data:
            raise KeyError("The app_config must have key 'application'.")
        if not isinstance( data, dict ):
            raise TypeError(f"The application information in application should be dict, but get {type(input_params)}")
        
        return data

    def _get_area_from_setting(self, app_setting: dict) -> list:
        areas  = app_setting.get("areas", None)
        if not areas:
            raise KeyError("Can not find areas key in application config.")
        if not isinstance(areas, list):
            raise TypeError("The expected type of the area setting shout be list.")
        return areas

    def _get_area_name(self, area_data: dict, default_name= "The defalt area") -> str:
        _name = area_data.get("name", None)

        if not _name:
            raise ValueError(f"Can not find 'name' in config.")

        elif _name == []:
            _name = default_name

        elif not isinstance(_name, str):
            raise TypeError("The type of area name should be str")

        return _name
    
    def _get_area_depend(self, area_data: dict, default_depend: list) -> list:
        # check has depend_on and type is correct
        _depend_on = area_data.get("depend_on", None)

        if not _depend_on:
            raise ValueError(f"Can not find 'depend_on' in area setting ")
        
        elif _depend_on==[]:
            _depend_on = default_depend

        elif not isinstance(_depend_on, list):
            raise TypeError("The type of area name should be list")

        return _depend_on

    def _get_norm_area_pts(self, area_data: dict, default_value: list= [[0,0],[1,0],[1,1],[0,1]]) -> list:
        """get normalized area point, if not setup means the whole area"""
        _area_point = area_data.get("area_point", None)

        if not _area_point:
            raise ValueError(f"Can not find 'area_point' in config.")
        
        elif _area_point == []:
            _area_point = default_value

        elif not isinstance(_area_point, list):
            raise TypeError("The type of area name should be list")

        return _area_point

    # NOTE: Init Event function
    def _get_event_obj(self, area_data: dict, event_func: EventHandler, drawer: DrawTool) -> Union[None, EventHandler]:
        events = area_data.get("events", None)
        if not events: return None 

        if not isinstance(area_data, dict):
            raise TypeError("Event setting should be dict.")
        
        event_obj = event_func(
            title = events["title"],
            folder = events.get("event_folder", "./events"),
            drawer = drawer,
            operator = events["logic_operator"],
            threshold = events["logic_value"],
            cooldown = events.get("cooldown", 1000)
        )
        return event_obj

    # NOTE: Get Area Setting function
    def get_areas_setting(self, app_setting: dict, labels: list) -> List[dict]:
        """Parse each area setting and generate a new area setting with new items.

        Args:
            app_setting (dict): the intput app setting ( params['application'] )
            labels (list): the labels object

        Returns:
            List[dict]: the new area setting that include new key ( norm_area_point, event )
        """
        # Parse each area data
        new_areas = []
        for area_data in self._get_area_from_setting(app_setting):
            new_areas.append({
                "name": self._get_area_name(area_data= area_data),
                "depend_on": self._get_area_depend(area_data= area_data, default_depend=labels),
                "norm_area_point": self._get_norm_area_pts(area_data= area_data),
                "event": self._get_event_obj(area_data= area_data, drawer= self.drawer, event_func= EventHandler)
            })
        
        return new_areas

    def update_all_area_point(self, frame:np.ndarray, areas: list) -> None:
        """ Denormalize all area point. 

        Args:
            frame (np.ndarray): the input frame
            areas (list): the normalize area point
        """
        # denormalize area point and add new key into areas    
        # get shape
        h, w = frame.shape[:2]
        if self.frame_h == h and self.frame_w == w:
            return

        self.frame_h, self.frame_w = h, w
        
        def map_wraper(area):
            denorm_area_pts = denorm_area_points( self.frame_w, self.frame_h, area["norm_area_point"])
            sorted_area_pts = sort_area_points(denorm_area_pts)
            area["area_point"] = sorted_area_pts

        # NOTE: map is faster than for loop
        # areas = list(map(map_wraper, areas))
        for area in areas:
            map_wraper(area)

        print("Updated area point !!!")
        
    def obj_in_area(self, px:int, py:int, poly:list) -> bool:
        """does the object in area or not

        Args:
            px (int): object center x
            py (int): object center y
            poly (list): area points

        Returns:
            bool: does object in area or not
        """
        is_in = False
        length = len(poly)

        for idx , corner in enumerate(poly):

            next_idx = idx + 1 if idx +1 < length else 0
            x1 , y1 = corner
            x2 , y2 = poly[next_idx]
            
            # if on the line
            if (x1 == px and y1 ==py) or (x2==px and y2 ==py):
                is_in = False
                break
            
            # if not in poly

            if min(y1,y2) <py <= max(y1 ,y2):
                
                x =x1+(py-y1)*(x2-x1)/(y2-y1)
                if x ==px:
                    is_in = False
                    break
                elif x > px:
                    is_in = not is_in
        
        return is_in

    def get_available_areas_and_update_output(self, label:str, xmin:int, ymin:int, xmax:int, ymax:int) -> list:
        """ parse all areas setting, check is label in depend_on and object in area,
        if available then update area["output"] 

        Args:
            label (str): the detected label
            xmin (int): the detected object position
            ymin (int): the detected object position
            xmax (int): the detected object position
            ymax (int): the detected object position

        Returns:
            list: the available areas
        """
        ret = []
        cnt_px, cnt_py = (xmin+xmax)//2, (ymin+ymax)//2
            
        for area_idx, area in enumerate(self.areas):
            area_pts = area["area_point"]
            depend = area["depend_on"]

            if label in depend and self.obj_in_area(cnt_px, cnt_py, area_pts):
                ret.append(area_idx)
                area["output"][label] += 1

        return ret

    # ------------------------------------------------

    def clear_app_output(self) -> None:
        """clear app output ( self.areas ) """
        for area in self.areas:
            area["output"] = defaultdict(int)

    # ------------------------------------------------
    # Drawing Meta Data ( Tracking data )

    def draw_event_data(self, frame:np.ndarray, event_data:dict) -> np.ndarray:
        """ Draw Tracking Data from meta data
        1. Update Area Points
        2. Bounding Box with Tracking ID.
        3. Application Output.
        4. Return Draw Image
        """
        
        self.drawer.update_draw_params(frame=frame)
        self.update_all_area_point(frame=frame, areas=self.areas)
        self.clear_app_output()

        overlay = frame
        
        # Combine all app output and get total object number
        detections = event_data["detections"]
        areas = [ event_data["area"] ]
        
        # Get Detections with tracked_idx
        for det in detections:
            [ xmin, ymin, xmax, ymax, label, score ] = \
                [ det[key] for key in [ "xmin", "ymin", "xmax", "ymax", "label", "score" ] ]

            color = self.drawer.get_color(label)

            self.drawer.draw_bbox(
                overlay, [xmin, ymin], [xmax, ymax], color )
            
            info = f"{label}: {int(score*100)}%"
            self.drawer.draw_label(overlay, info, [xmin, ymin], color )

        # Get Output
        self.drawer.draw_area_results(
            overlay, areas )

        return overlay

    # ------------------------------------------------

    @get_time
    def test(self,frame:np.ndarray, detections:list):
        return self.__call__(frame, detections)
    
    # NOTE: __call__ function is requirements
    def __call__(self, frame:np.ndarray, detections:list) -> Tuple[np.ndarray, list, list]:
        """
        1. Update basic parameters
        2. Parse detection and draw
        3. Check the event is trigger or not
        4. Return data
        """

        self.drawer.update_draw_params(frame=frame)
        self.update_all_area_point(frame=frame, areas=self.areas)
        self.clear_app_output()

        # Draw area
        original = frame.copy()
        overlay = frame
        
        if self.draw_area:
            overlay = self.drawer.draw_areas(overlay, self.areas )

        save_area_dets = defaultdict(list)

        # Draw inference results
        for det in detections:
            
            # get parameters
            label, score = det.label, det.score
            xmin, ymin, xmax, ymax = \
                tuple(map(int, [ det.xmin, det.ymin, det.xmax, det.ymax ] ))            
            
            # get available areas and update area["output"]
            available_areas = self.get_available_areas_and_update_output(
                label, xmin, ymin, xmax, ymax)
            
            # no available area
            if len(available_areas)==0: continue
            
            # Draw bounding box and label
            color = self.drawer.get_color(label)
            if self.draw_bbox:
                self.drawer.draw_bbox(
                    overlay, [xmin, ymin], [xmax, ymax], color )
            
            if self.draw_label:
                info = f"{label}: {int(score*100)}%"
                self.drawer.draw_label(overlay, info, [xmin, ymin], color )
            
            # Update area detection objects for saving meta data
            for area_idx in available_areas:
                save_area_dets[area_idx].append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "label": label,
                    "score": score,
                })

        # Draw app output
        if self.draw_result:
            self.drawer.draw_area_results(
                overlay, self.areas )

        # Combine all app output and get total object number
        app_output = []
        event_output = []
        for area_idx, area in enumerate(self.areas):
            
            total_obj_nums = 0

            area_output = dict(area["output"])

            # Current area has nothing
            if not area_output: continue

            # Re-Combine new area output for v1.1 version
            new_area_output = []
            for label, num in area_output.items():
                new_area_output.append({
                    "label": label,
                    "num": num
                })

            # Combine app_output
            app_output.append({
                "id": area_idx,
                "name": area["name"],
                "data": new_area_output 
            })

            # Event
            total_obj_nums += sum(area_output.values())

            event = area.get("event", None)
            if self.force_close_event or not event: continue
            
            cur_output = event( original= original,
                                value= total_obj_nums,
                                detections= save_area_dets[area_idx],
                                area= area )
            if not cur_output: continue

            event_output.append( cur_output )

        return overlay, {"areas": app_output}, {"event": event_output}

# ------------------------------------------------------------------------
# Test

def main():

    from argparse import ArgumentParser, SUPPRESS
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
                "car": [ 105, 125, 105 ],
                "truck": [ 125, 115, 105 ]
            },
            "areas": [ 
                {
                    "name": "Area0",
                    "depend_on": [
                        "car", "truck"
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
                        "uid":"",
                        "title": "Traffic in Area 1 is very heavy",
                        "logic_operator": ">",
                        "logic_value": 3,
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
                        "uid":"",
                        "title": "GGGGG",
                        "logic_operator": ">",
                        "logic_value": 3,
                    }
                }
            ]
        }
    }
    app = Detection_Zone(app_config ,args.label)

    # 7. Start Inference
    try:
        t_costs = []
        while True:
            # Get frame & Do infernece
            frame = src.read()
            
            results = model.inference(frame=frame)
          
            # frame , app_output , event_output = app(frame,results)
            (frame , app_output , event_output), t_cost = app.test(frame,results)
            t_costs.append(t_cost)
            if len(t_costs)==500:
                print('\n\n\n')
                print('Average Cost Time: ', round(sum(t_costs)/500*1000, 5))
                break
            
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
        print('Detected Key Interrupt !')

    finally:
        model.release()
        src.release()
        dpr.release()

if __name__=='__main__':
    """
    python3 apps/Detection_Zone.py \
    -m model/yolo-v3-tf/yolo-v3-tf.xml \
    -l model/yolo-v3-tf/coco.names \
    -i data/car.mp4 -at yolo
    """
    main()