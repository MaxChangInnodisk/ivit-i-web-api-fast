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
        self.cooldown = cooldown # Convert to nano second

        # Logic
        self.logic_map = get_logic_map()
        self.logic_event = self.logic_map[operator]
    
        # Timer
        self.trigger_time = None

        # Generate uuid
        self.folder = self.check_folder(folder)
        self.uuid = str(uuid.uuid4())[:8]
        self.uuid_folder = os.path.join(self.folder, self.uuid)
        
        # Dynamic Variable
        self.current_time = time.time_ns()
        self.current_value = None

        # Custom Threading Pool
        self.exec_pool = []
        self.pools = ThreadPool(processes=4)

        # Flag
        self.event_started = False

    def check_folder(self, folder_path: str) -> None:
        if not os.path.exists(folder_path):        
            os.makedirs(folder_path)
        return folder_path

    def is_trigger(self) -> bool:
        return self.logic_event(self.current_value , self.threshold)

    def is_ready(self) -> bool:

        if self.trigger_time is None:
            self.trigger_time = self.current_time     
            return True

        _time = (self.current_time - self.trigger_time)/1000000000
        if _time  < self.cooldown:
            print('\rNot ready, still have {} seconds'.format(round(self.cooldown-_time , 5)),end='')
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
        
        if self.event_started:
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
            "event_status": self.event_started,
            "meta": meta_data,
        }

    def reset(self) -> None:
        """ Reset timestamp and flag """
        self.start_time = None
        self.end_time = None
        self.event_started = False

    def _print_title(self, text: str):
        print(f'[EVENT] [{self.title.upper()}] {text}')

    def __call__(self, original: np.ndarray, value: int, detections: list, area: dict) -> Union[None, dict]:
        
        # Update variable
        self.current_time = time.time_ns()
        self.current_value = value
        
        # Trigger Event
        event_triggered = self.is_trigger()
        if event_triggered and self.event_started: 
            return  # None

        elif not event_triggered and not self.event_started:
            return self.reset() # None
        
        elif event_triggered and not self.event_started:
            self.start_time = self.current_time
        
            # Cooldown time: avoid trigger too fast
            if not self.is_ready(): return
            
        elif not event_triggered and self.event_started:
            self.end_time = self.current_time

        # Update status           
        self.event_started = event_triggered

        self._print_title('Event {} ( Total: {})'.format(
            'Start' if self.event_started else 'Stop', value))

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
