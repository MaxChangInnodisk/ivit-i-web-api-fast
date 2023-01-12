import os, sys, time, logging, math, cv2
import numpy as np 

sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp, CV_WIN
try:
    from DynamicBoundingBox import DynamicBoundingBox
    from Tracking import Tracking
    from MovingDirection import MovingDirection
    from TrackingArea import TrackingArea
except:
    from apps.DynamicBoundingBox import DynamicBoundingBox
    from apps.Tracking import Tracking
    from apps.MovingDirection import MovingDirection
    from apps.TrackingArea import TrackingArea

# Parameters
K_DETS          = "detections"
K_LABEL         = "label"
K_DEPEND        = "depend_on"
K_DRAW_BB       = "draw_bbox"
K_DRAW_INFO     = "draw_info"
K_DIS           = "distance"
K_BUFFER        = "buffer"
K_FRAME         = "wrong_frame"
K_ARROW_LEN     = "arrow_length"

K_DRAW_ARAE     = "draw_area"
K_DRAW_VEC      = "draw_vector"
K_AREA          = "area_points"
K_VECTOR        = "area_vector"

K_AREA_COLOR    = "area_color"
K_AREA_OPACITY  = "area_opacity"

K_SENS          = "sensitivity"
SENS_MED        = "medium"
SENS_HIGH       = "high"
SENS_LOW        = "low"

K_DEF_COLOR     = "default_color"
K_WARN_COLOR    = "warning_color"

K_ANGLE_THRES   = "angle_thres"


class WrongDirection( TrackingArea, MovingDirection, ivitApp):
    """ Wrong Direction Detect

    - Workflow
        1. Setup Area Points using `set_area`
        2. Setup Vector Points using `set_vector`
        3. Clear and compire each area and vector 
    """
    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')
        self.init_draw_params()
        
        self.init_area()
        self.init_area_vec_params()
        self.init_vector_params()
        self.init_info()

    def init_params(self):
        self.def_param( name=K_DEPEND, type='list', value=['car'] )
        self.def_param( name=K_DRAW_BB, type='bool', value=True )
        self.def_param( name=K_DRAW_VEC, type='bool', value=True)

        self.def_param( name=K_DIS, type='int', value=20, descr='The limit of tracking distance')
        self.def_param( name=K_DRAW_INFO, type='bool', value=True, descr="Display the app information on the top left corner.")
        self.def_param( name=K_BUFFER, type='int', value=20, descr='The buffer of the average vector')
        self.def_param( name=K_ARROW_LEN, type='float', value=20, descr='The length of the arrow')
        self.def_param( name=K_SENS, type='str', value=SENS_LOW )
        self.def_param( name=K_FRAME, type='int', value=5, descr="The times of the wrong vector frame")

        self.def_param( name=K_AREA, type='dict', value={} )
        self.def_param( name=K_VECTOR, type='dict', value={} )

        self.def_param( name=K_DEF_COLOR, type='tuple', value=(0,255,0) )
        self.def_param( name=K_WARN_COLOR, type='tuple', value=(0,0,255) )

        self.def_param( name=K_DRAW_ARAE, type='bool', value=True )
        self.def_param( name=K_AREA_COLOR, type='list', value=(0,0,255) )
        self.def_param( name=K_AREA_OPACITY, type='float', value=0.15 )

        self.def_param( name=K_ANGLE_THRES, type='int', value=90, descr='larger than this value will identify to error vector')

    def init_area_vec_params(self):
        """ Initialize the parameters of the vector in the area"""
        self.vec_pts = {}
        self.vec_pts_idx = 0   
        self.vec_draw_flag = False 

        # Track each object in which area
        self.cur_area = {}
        self.track_obj_area = {}
        
        # wrong vector
        self.wrong_vector = False
        self.wrong_vector_times = {}

    def draw_vector_event(self, frame):
        """ Draw ArrowLine from `self.vec_pts` """
        if len(self.vec_pts)==0 or not self.get_param(K_DRAW_ARAE): return frame

        for idx, pnts in self.vec_pts.items():
            if pnts==[]: continue
            cv2.arrowedLine(frame, tuple(pnts[0]), tuple(pnts[1]), self.get_param(K_AREA_COLOR), 2)
        return frame

    def cv_vector_handler(self, event, x, y, flags, param):
        
        org_img = param['img'].copy()   
        out = False

        if event == cv2.EVENT_LBUTTONDOWN and not self.vec_draw_flag and not out:
            # add first point if indext not exist
            if len(self.vec_pts[self.vec_pts_idx])>=2:
                self.vec_pts[self.vec_pts_idx].clear()

            self.vec_pts[self.vec_pts_idx].append( [x,y] )

            self.vec_draw_flag = True
            cv2.circle(org_img, (x, y), 3, (0, 0, 255),5)
            cv2.imshow(CV_WIN, org_img)
            
        elif event == cv2.EVENT_LBUTTONUP and self.vec_draw_flag and not out:
            
            # Add Arrow
            self.vec_pts[self.vec_pts_idx].append( [x,y] ) 
            
            # Draw
            cv2.arrowedLine(org_img, tuple(self.vec_pts[self.vec_pts_idx][0]), tuple(self.vec_pts[self.vec_pts_idx][1]), (0,0,255), 2)
            cv2.imshow(CV_WIN, org_img)

            # Update Information
            # self.vec_pts_idx      += 1
            self.vec_draw_flag   = False
            out = True

        elif event == cv2.EVENT_MOUSEMOVE and self.vec_draw_flag and not out:

            image = org_img.copy()
            cv2.line(image, tuple(self.vec_pts[ self.vec_pts_idx][0]), (x, y), (0,0,0), 1)
            cv2.imshow(CV_WIN, image)

    def check_area_vector_exist(self):
        if None in [ self.vec_pts_idx, self.vec_pts, self.area_pts_idx, self.area_pts ]:
            return False
        return True

    def check_area_vector(self):

        def show_content(dictionary, title="Object", tab='\t'):
            _tab = '\t'
            print(tab, title)
            for key, val in dictionary.items():
                print("{} [{}] : {}".format(tab+_tab, key, val))
        
        def show_both(title="This is title", tab=""):
            _tab = '\t'
            print(tab, title)
            show_content(self.area_pts, "Area Point", tab+_tab)
            show_content(self.vec_pts, "Area Vector", tab+_tab)

        if not self.check_area_vector_exist(): return None

        show_both("Origin: ")

        # Real Data Length
        diff_list       = []
        arrow_list      = list(self.vec_pts.keys())
        area_point_list = list(self.area_pts.keys())
        
        # Get Diff Value in Two List
        for key in area_point_list:
            if not(key in arrow_list):
                diff_list.append(key)
            else:
                arrow_list.remove(key)
        [ diff_list.append(key) for key in arrow_list ]
        
        # Try to remove
        for key in diff_list:
            try: self.area_pts.pop(key)
            except Exception as e: pass
            try: self.vec_pts.pop(key)
            except Exception as e: pass

        show_both("Get paired data: ")

        # -----------------------------------------
        
        # Clear Pair but Empty Content
        area_point_copy = self.area_pts.copy()
        for key in area_point_copy.keys():
            
            if ( [] in [ self.area_pts[key], self.vec_pts[key] ] ):
                self.area_pts.pop(key)
                self.vec_pts.pop(key)
        
        self.area_pts_idx =  len(self.area_pts)-1
        self.vec_pts_idx      = len(self.vec_pts) -1
        
        show_both("Get available data: ")

    def set_area(self, frame=None):
        
        if frame.all()==None:
            if self.area_pts != {} and self.vec_pts != {}:
                return True
            msg = "Could not capture polygon coordinate and there is not provide frame to draw"
            raise Exception(msg)

        if self.area_pts != {} and self.vec_pts != {}:
            logging.info("Detected area point is already exist")
            
        # Prepare Image to Draw
        logging.info("Detected frame, open the cv windows to collect area coordinate")

        # Initialize CV Windows
        cv2.namedWindow(CV_WIN,cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(CV_WIN,cv2.WND_PROP_ASPECT_RATIO,cv2.WINDOW_KEEPRATIO)

        # Setup Mouse Event and Display
        while(True):
            
            # Draw Current Area and Vector
            temp_frame = frame.copy()
            temp_frame = self.draw_area_event( temp_frame )
            temp_frame = self.draw_vector_event( temp_frame )

            # Init: Update Index
            self.area_pts_idx += 1                
            self.vec_pts_idx += 1
            self.area_pts[self.area_pts_idx] = list()
            self.vec_pts[self.vec_pts_idx] = list()
            
            # Set up area handler
            cv2.setMouseCallback(CV_WIN, self.cv_area_handler, {"img": temp_frame})
            cv2.putText(temp_frame, "Click to define detected area, press any key to leave", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow(CV_WIN, temp_frame)
            key = cv2.waitKey(0)

            # For Direction
            temp_frame = frame.copy()
            temp_frame = self.draw_area_event( temp_frame )
            temp_frame = self.draw_vector_event( temp_frame )
            cv2.setMouseCallback(CV_WIN, self.cv_vector_handler, {"img": temp_frame})
            cv2.putText(temp_frame, "Click to define direction, press q to leave, other to continue", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow(CV_WIN, temp_frame)
            
            key2 = cv2.waitKey(0)
            if key2 in [ ord('q'), 27 ]: break
            else: continue

        cv2.setMouseCallback(CV_WIN, lambda *args : None)
        cv2.destroyAllWindows()

        self.check_area_vector()

    def init_info(self):
        super().init_info()
        self.wrong_vec_num = {}

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

        self.cur_area[label] = list()
        self.track_obj_area[label]=dict()

    def update_point_and_draw(self, info ):
        """
        Capture all center point in current frame and draw the bounding box

        1. Add detected object into `self.detected_labels`
        2. Store the center point of each object in `self.cur_pts`
        3. If the first frame, add `cur_pt` into `track_obj` directly.
        """

        for det in info[K_DETS]:

            # Get Detection Object
            label = det[K_LABEL]
            
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
            self.cur_area[label].append( cur_in_area )

            # Add into track_obj at first time
            if self.frame_idx <= 1: 
                self.track_obj[label][ self.track_idx[label] ] = cur_pt
                self.track_obj_bbox[label][ self.track_idx[label] ] = (x1,x2,y1,y2)
                self.track_obj_area[label][ self.track_idx[label] ] = cur_in_area

                self.track_idx[label] +=1

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
            __cur_area = self.cur_area[label].copy()
            
            for track_idx, prev_pt in __track_obj.items():
                
                obj_exist = False

                for cur_idx, cur_pt in enumerate(__cur_pts):                
                    
                    # Lower than the limit of distance means that's not the same object, continue to the next
                    if ( self.get_distance(cur_pt, prev_pt) > self.get_param(K_DIS) ):
                        continue

                    # Step 1
                    self.track_obj[label][track_idx] = cur_pt
                    self.track_obj_bbox[label][track_idx] = __cur_bbox[cur_idx]
                    self.track_obj_area[label][track_idx] = __cur_area[cur_idx]
                    
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
                    self.track_obj_area[label].pop(track_idx, None)

    # Re-define the method for vector detection
    def track_new_object_and_draw(self, frame):
        """
        Track new object after `track_prev_object` function

        1. The remain point in `self.cur_pts` can identify to the new object
        2. Calculate the vector.
            a. Update direction in `self.track_obj_vec`
        """
        
        # update wrong_vec_num 
        for area_idx in self.area_pts.keys():
            self.wrong_vec_num.update( {area_idx: dict()} )

        # traval each label
        for label in self.detected_labels:
            
            # update track object
            for new_idx, new_pt in enumerate(self.cur_pts[label]):
                self.track_obj[label][ self.track_idx[label] ] = new_pt
                self.track_obj_bbox[label][ self.track_idx[label] ] = self.cur_bbox[label][new_idx]
                self.track_obj_area[label][ self.track_idx[label] ] = self.cur_area[label][new_idx]

                self.track_idx[label] +=1

            # draw the track number on object
            self.wrong_vector = False
            for track_idx, cur_pt in self.track_obj[label].items():

                # Check in Area
                in_area = False
                for area_idx, area_pt in self.area_pts.items():
                    if self.in_poly( pts=area_pt, pts_list=cur_pt ):
                        in_area = True
                if not in_area: 
                    continue

                # Get Vector ( Arrow )
                if not self.update_vector( label, track_idx ): 
                    continue                
                vector = self.track_obj_vec[label].get(track_idx)
                
                # Get angle
                obj_vec_angle = self.get_angle_for_cv(vector[1], vector[0])
                area_vec_angle = self.get_angle_for_cv(self.vec_pts[in_area][1], self.vec_pts[in_area][0])

                # Check the vector is correct
                
                color = self.get_param(K_DEF_COLOR)
                    
                if abs( obj_vec_angle - area_vec_angle ) < self.get_param(K_ANGLE_THRES):

                    # Update `self.wrong_vector_times`
                    if self.wrong_vector_times.get(label) is None:
                        self.wrong_vector_times.update( {label:0} )
                    self.wrong_vector_times[label]+=1

                    # Really Wrong Vector
                    if self.wrong_vector_times[label] >= self.get_param(K_FRAME):
                        
                        # Clear wrong vector
                        self.wrong_vector_times[label] = 0
                        
                        # Update color
                        color = self.get_param(K_WARN_COLOR)
                        self.wrong_vector = True

                        # Tracking object number in area
                        cur_area_idx = self.track_obj_area[label][track_idx]
                        if self.wrong_vec_num[cur_area_idx].get(label) is None:
                            self.wrong_vec_num[cur_area_idx].update({label:0})
                        self.wrong_vec_num[cur_area_idx][label]+=1
                else:
                    if self.wrong_vector_times.get(label) is not None:
                        self.wrong_vector_times.pop(label)

                # Draw Bounding Box
                if self.get_param(K_DRAW_BB):
                    x1,x2,y1,y2 = self.track_obj_bbox[label][track_idx]
                    cv2.rectangle(frame, (x1,y1), (x2, y2), color, self.thick)

                # Draw Arrow
                if self.get_param(K_DRAW_VEC):
                    cv2.arrowedLine(
                        frame, vector[1], vector[0],
                        color, self.thick+1, tipLength = 0.5   )

        # Update alarm information
        if self.wrong_vector:
            self.alarm = ', '.join([ 'Area {} : {}'.format(area_idx, ', '.join([ f'{_num} {_label}' for _label, _num in area_num.items() ])) \
                for area_idx, area_num in self.wrong_vec_num.items() ])
        else:
            self.alarm = ''

        return frame

    def draw_app_info(self, frame):
        if self.get_param(K_DRAW_INFO) and self.alarm!='':
            (t_wid, t_hei), t_base = cv2.getTextSize(self.alarm, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = 10, 10, 10+t_wid, 10+t_hei+t_base
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), (0,0,255) , -1)
            cv2.putText(
                frame, self.alarm, (t_xmin, t_ymax-t_base), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )

    def run(self, frame, data, draw=True) -> tuple:
        """
        Define the worflow when execute the application

        1. Tracking Object
        2. Record the vector between previous and current object
        3. Define the mount of the buffer to stablizing the final vector
        """
        if data is None:
            return frame, data
            
        # Init
        self.frame_idx += 1
        self.clear_current_point()
        self.update_draw_param( frame )

        # Draw Area and Vector
        frame = self.draw_area_event(frame)
        frame = self.draw_vector_event(frame)

        # Capture all center point in current frame and draw the bounding box
        self.update_point_and_draw( data )
        
        # if not first frame: start to calculate distance to check the object is the same one
        self.track_prev_object()

        # adding the remaining point to track_obj
        self.track_new_object_and_draw( frame )

        # get tracking information which will be stored in `self.app_info`
        info = self.draw_app_info( frame )

        return ( frame, self.wrong_vec_num)

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
        'draw_info': True,
        'draw_bbox': True,
        'draw_info': False,
        'draw_vector': False,
        'distance': 50,
        "buffer": 50,
        'wrong_frame': 3
    }

    ivit = get_ivit_model(model_type)
    ivit.set_async_mode()
    ivit.load_model(model_conf)
    
    # Def Application
    app = WrongDirection(params=app_conf, label=model_conf['openvino']['label_path'])
    
    # Get Source
    data, fps, fps_pool = None, -1, []

    source = './data/car.mp4'
    # source = './data/wrong-side.mp4'
    cap = cv2.VideoCapture(source)
    
    ret, frame = cap.read()
    app.set_area(frame)
    
    while(cap.isOpened()):

        t_start = time.time()

        ret, frame = cap.read()
        
        if not ret: 
            cap = cv2.VideoCapture(source)
            time.sleep(1)
            continue

        draw = frame.copy()

        _data = ivit.inference(frame=frame)
        data = _data if _data else data

        draw, info = app(draw, data)
        
        print(info)

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
        elif press_key == ord('v'): 
            app.set_param('draw_vector', not app.get_param('draw_vector')) 
        elif press_key == ord('i'):
            app.set_param('draw_info', not app.get_param('draw_info'))
        elif press_key in [ord('+'), ord('=')]:
            app.set_param('arrow_length', app.get_param('arrow_length')+1 )
        elif press_key in [ord('-'), ord('_')]:
            app.set_param('arrow_length', app.get_param('arrow_length')-1 )

        # Calculate FPS
        if _data:
            fps_pool.append( int(1/(time.time()-t_start)) )
            if len(fps_pool)>10:
                fps = sum(fps_pool)//len(fps_pool)
                fps_pool.clear()

    cap.release()
    ivit.release()
