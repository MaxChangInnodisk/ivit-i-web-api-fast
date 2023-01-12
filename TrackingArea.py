import sys, os, logging, math, cv2
import numpy as np
import itertools as it

sys.path.append( os.getcwd() )
from ivit_i.utils.err_handler import handle_exception

from ivit_i.app.common import (
    ivitApp, 
    DETS,
    LABEL,
    CV_WIN,
    get_time,
 )

from ivit_i.utils.draw_tools import ( draw_rect )

try:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    from Tracking import Tracking
    from AreaDetection import AreaDetection
except Exception as e:
    from apps.Tracking import Tracking
    from apps.AreaDetection import AreaDetection

DETS            = "detections"
LABEL           = "label"

K_DEPEND        = "depend_on"
K_DRAW_BB       = "draw_bbox"
K_DRAW_ARAE     = "draw_area"
K_DRAW_INFO     = "draw_info"
K_AREA          = "area_points"

K_STAY_TIME     = "stay_time"
K_STAY_THRES    = "stay_frame_thres"

K_SAVE_IMG      = "save_image"
K_SAVE_FOLDER   = "save_folder"

K_ALARM         = "alarm"
K_T_ALARM       = "alarm_time"

K_AREA_COLOR    = "area_color"
K_AREA_OPACITY  = "area_opacity"

K_SENS          = "sensitivity"
SENS_MED        = "medium"
SENS_HIGH       = "high"
SENS_LOW        = "low"

K_DIS = "distance"

AREA_COLOR = (0,0,255)
ARAE_OPACITY = 0.3
AREA_PT = "area_point"
ALPHA = 0.5

class TrackingArea(Tracking, AreaDetection, ivitApp):

    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')

        # Track
        self.init_track_params()
        self.init_draw_params()
        self.init_info()
        self.init_area()
        
    def init_params(self):
        self.def_param( name=K_DEPEND, type='list', value=['car'] )
        self.def_param( name=K_DRAW_BB, type='bool', value=True )
        self.def_param( name=K_AREA, type='dict', value={} )
        self.def_param( name=K_DRAW_ARAE, type='bool', value=True )
        self.def_param( name=K_AREA_COLOR, type='list', value=(0,0,255) )
        self.def_param( name=K_AREA_OPACITY, type='float', value=0.15 )
        self.def_param( name=K_SENS, type='str', value=SENS_MED )
        self.def_param( name=K_STAY_TIME, type='int', value=3, descr='the staying time to trigger')
        self.def_param( name=K_STAY_THRES, type='int', value=0.9, descr='the threshold of the staying object to determine the object is actually enter the area.')
        self.def_param( name=K_SAVE_IMG, type='bool', value=True, descr="bool option for save image" )
        self.def_param( name=K_SAVE_FOLDER, type='str', value='data', descr="define the saved image folder")        
        self.def_param( name=K_ALARM, type='str', value='', descr="define alarm content")
        self.def_param( name=K_T_ALARM, type='int', value=3, descr="display alarm alive time")
        self.def_param( name=K_DIS, type='int', value=60, descr='The limit of tracking distance')
        self.def_param( name=K_DRAW_INFO, type='bool', value=True, descr="Display the app information on the top left corner.")

    def update_point_and_draw(self, info ):
        """
        Capture all center point in current frame and draw the bounding box

        1. Add detected object into `self.detected_labels`
        2. Store the center point of each object in `self.cur_pts`
        3. If the first frame, add `cur_pt` into `track_obj` directly.
        """

        for det in info[DETS]:

            # Get Detection Object
            label = det[LABEL]
            
            # Pass the label we don't want
            if not (self.depend_label(label, self.get_param(K_DEPEND))): 
                continue

            # Parse Bounding Box Infor
            [ x1, x2, y1, y2 ] = [ det[key] for key in [ 'xmin', 'xmax', 'ymin', 'ymax'] ]
            
            # Get Center and Saving the center point
            cur_pt = ((x1 + x2)//2, (y1 + y2)//2)

            # if not in area, then jump to the next
            cur_in_area = self.check_obj_area( (x1, x2, y1, y2) )
            if cur_in_area == (-1): continue

            # Update detected_labels and etc...
            self.update_track_param(label)

            self.cur_pts[label].append( cur_pt )
            self.cur_bbox[label].append( (x1,x2,y1,y2) )

            # Add into track_obj at first time
            if self.frame_idx <= 1: 
                self.track_obj[label][ self.track_idx[label] ] = cur_pt
                self.track_obj_bbox[label][ self.track_idx[label] ] = (x1,x2,y1,y2)
                self.track_idx[label] +=1

    def run(self, frame, data, draw=True) -> tuple:
        """
        Define the worflow when execute the application

        1. Capture all center point in current frame and draw the bounding box if need
        2. Calculate the distance and store in variable
        """
        if data is None:
            return frame, data

        # Draw Area
        frame = self.draw_area_event(frame)

        # Init
        self.frame_idx += 1
        self.clear_current_point()
        self.update_draw_param( frame )

        # Capture all center point in current frame and draw the bounding box
        self.update_point_and_draw( data )
        
        # if not first frame: start to calculate distance to check the object is the same one
        self.track_prev_object()

        # adding the remaining point to track_obj
        self.track_new_object_and_draw( frame )

        # get tracking information which will be stored in `self.app_info`
        info = self.draw_app_info( frame )

        return ( frame, info)

if __name__ == "__main__":

    import cv2, time
    from ivit_i.common.model import get_ivit_model

    FONT, FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 1, 1

    # Define iVIT Model
    model_type = 'obj'
    model_conf = { 
        "tag": model_type,
        "openvino": {
            "model_path": "./model/yolo-v3-tf/FP32/yolo-v3-tf.xml",
            "label_path": "./model/yolo-v3-tf/coco.names",
            "anchors": [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 ],
            "architecture_type": "yolo",
            "device": "CPU",
            "thres": 0.4
        }
    }

    app_conf = {
        'depend_on': [ 'car' ],
        'draw_bbox': False,
        'draw_area': True,
        'sensitivity': 'medium'   }

    ivit = get_ivit_model(model_type)
    ivit.set_async_mode()
    ivit.load_model(model_conf)
    
    # Def Application
    app = TrackingArea(
        params=app_conf, 
        label=model_conf['openvino']['label_path']
    )
    
    # Get Source
    data, fps, fps_pool = None, -1, []

    cap = cv2.VideoCapture('./data/car.mp4')
    
    ret, frame = cap.read()
    app.set_area(frame)
    
    while(cap.isOpened()):

        t_start = time.time()

        ret, frame = cap.read()
        draw = frame.copy()
        if not ret: break

        # Inference
        infer_data = ivit.inference(frame=frame)
        if not infer_data is None:
            data = infer_data

        # Application
        draw, info = app(draw, data)
        
        cv2.putText( draw, f"FPS: {fps}", (draw.shape[1]-200, 40), cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, (0,0,255), FONT_THICKNESS, FONT )
        
        cv2.imshow('Tracking Sample', draw)

        press_key = cv2.waitKey(1) 
        if press_key in [ ord('q'), 27 ]: 
            break
        elif press_key == ord('b'): 
            app.set_param('draw_bbox', not app.get_param('draw_bbox')) 
        elif press_key == ord('a'): 
            app.set_param('draw_area', not app.get_param('draw_area')) 
        elif press_key == ord('i'):
            app.set_param('draw_info', not app.get_param('draw_info'))
        elif press_key in [ ord('='), ord('+') ]:
            app.set_param('distance', app.get_param('distance')+5)
        elif press_key in [ ord('-'), ord('_') ]:
            app.set_param('distance', app.get_param('distance')-5)

        # Calculate FPS
        if infer_data:
            fps_pool.append( int(1/(time.time()-t_start)) )
            if len(fps_pool)>10:
                fps = sum(fps_pool)//len(fps_pool)
                fps_pool.clear()

    cap.release()
    ivit.release()
