import sys, os, logging, math, cv2
import numpy as np
import itertools as it

sys.path.append( os.getcwd() )
from ivit_i.utils.err_handler import handle_exception

from ivit_i.app.common import (
    ivitApp, 
    DETS,
    LABEL,
    get_time,
    format_time, 
    parse_delta_time )

try:
    from DynamicBoundingBox import DynamicBoundingBox
except:
    from apps.DynamicBoundingBox import DynamicBoundingBox

K_DETS = "detections"
K_LABEL = "label"

K_DEPEND = "depend_on"
K_DRAW_BB = "draw_bbox"
K_DRAW_INFO = "draw_info"

K_DIS = "distance"

class Tracking(DynamicBoundingBox, ivitApp):

    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')

        self.init_draw_params()

        # Track
        self.init_track_params()
        self.init_info()

    def init_params(self):
        self.def_param( name=K_DEPEND, type='list', value=['car'] )
        self.def_param( name=K_DRAW_BB, type='bool', value=True )
        self.def_param( name=K_DIS, type='int', value=50, descr='The limit of tracking distance')
        self.def_param( name=K_DRAW_INFO, type='bool', value=True, descr="Display the app information on the top left corner.")

    def init_info(self):
        self.total_num  = dict()
        self.alarm      = ""
        self.app_info   = ""
        self.app_time   = get_time( False )

    def init_track_params(self):
        """ Initialize Tracking Parameters """
        self.detected_labels = []
        self.cur_pts    = {}
        self.cur_bbox   = {}
        self.track_obj      = {}
        self.track_idx      = {}
        self.track_obj_bbox = {}

    def update_track_param(self, label):
        """ Update Tracking Parameters with new label

        Update label key in `self.detected_labels`, `self.cur_pts`, `self.track_obj`, `self.track_idx`
        """
        if ( label in self.detected_labels ):
            return None

        self.detected_labels.append(label)
        self.cur_pts[label]=list()
        self.track_obj[label]=dict()
        self.track_idx[label]=0

        self.cur_bbox[label] = list()
        self.track_obj_bbox[label]=dict()

    def clear_current_point(self):
        """ Clear each label point in `self.cur_pts` """
        [ self.cur_pts[key].clear() for key in self.cur_pts ]
        [ self.cur_bbox[key].clear() for key in self.cur_bbox ]

    def update_point_and_draw(self, info ):
        """
        Capture all center point

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
            
            # Update detected_labels and etc...
            self.update_track_param(label)

            # Parse Bounding Box Infor
            [ x1, x2, y1, y2 ] = [ det[key] for key in [ 'xmin', 'xmax', 'ymin', 'ymax'] ]
            
            # Get Center and Saving the center point
            cur_pt = ((x1 + x2)//2, (y1 + y2)//2)
            self.cur_pts[label].append( cur_pt )

            # Store the bounding box
            self.cur_bbox[label].append( (x1,x2,y1,y2) )

            # Add into track_obj at first time
            if self.frame_idx <= 1: 
                self.track_obj[label][ self.track_idx[label] ] = cur_pt
                self.track_obj_bbox[label].update( {self.track_idx[label]: (x1,x2,y1,y2)} )
                self.track_idx[label] +=1

    def get_distance(self, pt, pt2):
        """ Calculate Euclidean Distance """
        pt, pt2 = ( float(pt[0]), float(pt[1]) ), ( float(pt2[0]), float(pt2[1]) )
        return math.hypot( pt2[0]-pt[0], pt2[1]-pt[1])

    def track_prev_object(self):
        """
        Tracking Object by counting each point's distance in `cur_pt` and `track_obj` ( the previous one )

        - If the distance lower than limit ( `self.distance` ) then it's the same object
            - add center point into `track_obj` ( Step 1 )
            - remove center point in `cur_pt` ( Step 2 )
            - set `obj_exist` to `True` ( Step 3 )

        - If not exist means maybe the object is disappear.
            - pop out from `track_obj` ( Step 4 )
        """
        
        for label in self.detected_labels:

            __track_obj  = self.track_obj[label].copy()
            __cur_pts = self.cur_pts[label].copy()
            __cur_bbox = self.cur_bbox[label].copy()
            
            for track_idx, prev_pt in __track_obj.items():
                
                obj_exist = False

                for cur_idx, cur_pt in enumerate(__cur_pts):                
                    
                    # Lower than the limit of distance means that's not the same object, continue to the next
                    if ( self.get_distance(cur_pt, prev_pt) > self.get_param(K_DIS) ):
                        continue

                    # Step 1
                    self.track_obj[label][track_idx] = cur_pt
                    self.track_obj_bbox[label][track_idx] = __cur_bbox[cur_idx]
                    
                    # Step 2
                    if cur_pt in self.cur_pts[label]:
                        self.cur_pts[label].remove(cur_pt)
                        self.cur_bbox[label].remove( __cur_bbox[cur_idx] )
                        
                    # Step 3
                    obj_exist = True
                
                # Step 4
                if not obj_exist:
                    self.track_obj[label].pop(track_idx)
                    self.track_obj_bbox[label].pop(track_idx, None) 

    def track_new_object_and_draw(self, frame, draw=True):
        """
        Track new object after `track_prev_object` function

        The remain point in `self.cur_pts` can identify to the new object
        """

        detected_list = []
        
        for label in self.detected_labels:

            # update track object
            for new_idx, new_pt in enumerate(self.cur_pts[label]):
                self.track_obj[label][ self.track_idx[label] ] = new_pt
                self.track_obj_bbox[label][ self.track_idx[label] ] = self.cur_bbox[label][new_idx]
                self.track_idx[label] +=1
            
            cur_total_num = list(self.track_obj[label].keys())
            
            # update key
            if not label in self.total_num: 
                self.total_num.update( { label: 1 } )

            # update total number
            self.total_num[label] = cur_total_num[-1] if cur_total_num != [] else self.total_num[label]
            
            # add each label alarm 
            detected_list.append( f"{self.total_num[label]+1} {label}" )
            
            # draw the track number on object
            if (not draw) or (frame is None): continue
            for idx, pt in self.track_obj[label].items():
                
                # Information
                track_info = str(f'{idx+1:03}')
                (t_wid, t_hei), t_base = cv2.getTextSize(track_info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
                half_wid, half_hei = t_wid//2, t_hei//2
                [cnt_x, cnt_y] = pt
                t_xmin, t_ymin, t_xmax, t_ymax = cnt_x-half_wid, cnt_y-half_hei, cnt_x+half_wid, cnt_y+half_hei
                cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), self.get_color(label) , -1)
                cv2.putText(
                    frame, track_info, (t_xmin, t_ymin+t_hei), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
                )

                # Draw Bounding Box
                if self.get_param(K_DRAW_BB):
                    x1,x2,y1,y2 = self.track_obj_bbox[label][idx]
                    cv2.rectangle(frame, (x1,y1), (x2, y2), self.get_color(label), self.thick)
        
        # Get the number of each object
        self.alarm = ', '.join( detected_list )

        return frame

    def draw_app_info(self, frame):
        """ Combine the information of the application. """
        
        # Get Current Time
        self.app_cur_time = get_time( False )
        self.live_time = parse_delta_time((self.app_cur_time-self.app_time))
        
        # Live Time for Display
        ret_live_time = "{}:{}:{}:{}".format(
            self.live_time["day"], 
            self.live_time["hour"], 
            self.live_time["minute"], 
            self.live_time["second"] )

        # Combine the result
        self.app_info = {
            "start"     : format_time(self.app_time),
            "current"   : format_time(self.app_cur_time),
            "duration"  : ret_live_time,
            "alarm"     : self.alarm
        }

        # Draw Application Information
        app_info = "Live Time: {} Detected: {}".format(ret_live_time, self.alarm)
        
        if self.get_param(K_DRAW_INFO):
            (t_wid, t_hei), t_base = cv2.getTextSize(app_info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = 10, 10, 10+t_wid, 10+t_hei+t_base
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), (0,0,255) , -1)
            cv2.putText(
                frame, app_info, (t_xmin, t_ymax-t_base), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )

        return app_info

    def run(self, frame, data, draw=True) -> tuple:
        """
        Define the worflow when execute the application

        1. Capture all center point in current frame and draw the bounding box if need
        2. Calculate the distance and store in variable
        """
        if data is None:
            return frame, data
            
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
            "thres": 0.9
        }
    }

    app_conf = {
        'depend_on': [],
        'draw_bbox': True,
        'draw_info': False
    }

    ivit = get_ivit_model(model_type)
    ivit.set_async_mode()
    ivit.load_model(model_conf)
    
    # Def Application
    app = Tracking(params=app_conf, label=model_conf['openvino']['label_path'])

    # Get Source
    data, fps, fps_pool = None, -1, []

    cap = cv2.VideoCapture('./data/car.mp4')
    while(cap.isOpened()):

        t_start = time.time()

        ret, frame = cap.read()
        draw = frame.copy()
        if not ret: break

        _data = ivit.inference(frame=frame)
        data = _data if _data else data

        draw, info = app(draw, data)
        
        cv2.putText( draw, f"FPS: {fps}", (draw.shape[1]-200, 40), cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, (0,0,255), FONT_THICKNESS, FONT )
        
        cv2.imshow('Tracking Sample', draw)

        press_key = cv2.waitKey(1) 
        if press_key in [ ord('q'), 27 ]: 
            break
        elif press_key == ord('b'): 
            app.set_param('draw_bbox', not app.get_param('draw_bbox')) 
        elif press_key == ord('i'):
            app.set_param('draw_info', not app.get_param('draw_info'))
        elif press_key in [ord('+'), ord('=')]:
            cur_dis = app.get_param('distance')
            print('Reduce Sensitivity :{}'.format(int(cur_dis)+1))
            app.set_param('distance', cur_dis+1 )
        elif press_key in [ord('-'), ord('_')]:
            cur_dis = app.get_param('distance')
            print('Increase Sensitivity :{}'.format(int(cur_dis)-1))
            app.set_param('distance', cur_dis-1 )

        # Calculate FPS
        if _data:
            fps_pool.append( int(1/(time.time()-t_start)) )
            if len(fps_pool)>10:
                fps = sum(fps_pool)//len(fps_pool)
                fps_pool.clear()

    cap.release()
    ivit.release()
