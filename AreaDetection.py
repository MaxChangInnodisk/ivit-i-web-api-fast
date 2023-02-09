import sys, os, logging, math, cv2
import numpy as np
import itertools as it

# iVIT_I
sys.path.append( os.getcwd() )
from ivit_i.utils.err_handler import handle_exception
from ivit_i.app.common import ivitApp, get_time, CV_WIN

# Inherit Class
try:
    from DynamicBoundingBox import DynamicBoundingBox
except:
    from apps.DynamicBoundingBox import DynamicBoundingBox

# AI Result
DETS            = "detections"
LABEL           = "label"

# Parameters
K_DEPEND        = "depend_on"
K_AREA          = "area_points"
K_SENS          = "sensitivity"

# Draw Option
K_DRAW_BB       = "draw_bbox"
K_DRAW_ARAE     = "draw_area"

# Draw Area Option
K_AREA_COLOR    = "area_color"
K_AREA_OPACITY  = "area_opacity"




# Define Sensitive Map: [ trigger number, on the border ]
# if trigger_number set 3, it will generate 3*3 = 9 
SENS_MED        = "medium"
SENS_HIGH       = "high"
SENS_LOW        = "low"
SENS_MAP = {
    SENS_LOW: [1, False],
    SENS_MED: [2, False],
    SENS_HIGH: [3, True]
}

class AreaDetection(DynamicBoundingBox, ivitApp):
    """ Area Detection
    The application could detect the object enter the area
    1. Check `area_points` is available with `self.check_area_pts`
    2. Get Trigger 
    """

    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')

        # Init Parent Function
        self.init_draw_params() # From DynamicBoundingBox
        
        # Update Parameter
        self.init_info()
        self.init_area()
        self.last_frame = { }

    def init_params(self):
        self.def_param( name=K_DEPEND, type='list', value=['car'] )
        self.def_param( name=K_AREA, type='dict', value={} )
        self.def_param( name=K_SENS, type='str', value=SENS_LOW )

        self.def_param( name=K_DRAW_BB, type='bool', value=True )
        self.def_param( name=K_DRAW_ARAE, type='bool', value=True )
        self.def_param( name=K_AREA_COLOR, type='list', value=(0,0,255) )
        self.def_param( name=K_AREA_OPACITY, type='float', value=0.15 )

    def check_area_pts(self):
        """ Clear un-available and un-paired area points """
        _area_pts = self.area_pts.copy()
        for idx, pts in _area_pts.items():
            # Pop out empty, [] data in area_points
            if ( pts == None ) or ( len(pts) == 0 ):
                self.area_pts.pop(idx)
                logging.warning('Clear unavailable data: {}'.format(idx, pts))

    def init_area(self):
        """ Parsing `self.area_pts` and `self.area_pts_idx` from `self.get_param`
        1. `self.area_pts`    
            - type: dict
            - desc: the area point is a dictionary which key is the index of the area, value is a list of area point ( x, y )
            - exam: { '0': [ (x1,y1), (x2,y2) ], '1': [ (x1,y1), (x2,y2) ] }
        2. `self.area_pts_idx`:
            - type: int
            - desc: the last index of the area point, if there has 3 area, then the key maybe is 2.
            - exam: 1
        """

        # Init Area Center
        self.area_cnt = {}
        self.area_obj_num = {}

        # Get Area Points
        self.area_pts = self.get_param(K_AREA)

        # Get Area Points Key
        area_points_key_list = [0] if self.area_pts == {} \
            else list(self.area_pts.keys())
        area_points_key_list.sort()
        self.area_pts_idx = area_points_key_list[-1]
        
        # Check Available Area Points
        self.check_area_pts()

        # logging.info('Get current area point index: {}'.format(self.area_pts_idx))
        # logging.info('Get area: {}'.format(self.area_pts))

    def draw_area_event(self, frame, area_color=None, area_opacity=None, in_cv_event=False, draw_points=False, draw_polys=True):
        """ Draw Detecting Area and update center point if need.
        - args
            - frame: input frame
            - area_color: control the color of the area
            - area_opacity: control the opacity of the area
        """
        
        # Get Parameters
        area_color = self.get_param(K_AREA_COLOR) if area_color is None else area_color
        area_opacity = self.get_param(K_AREA_OPACITY) if area_opacity is None else area_opacity
        
        #FIXME: Change to bitwise and store the template at first to speed up 
        # Parse All Area
        overlay = frame.copy()
        for area_idx, area_pts in self.area_pts.items():
            
            if area_pts==[]: break

            # Draw
            if self.get_param(K_DRAW_ARAE):
                # Draw circle if need
                if draw_points: 
                    [ cv2.circle(frame, tuple(pts), 3, area_color, -1) for pts in area_pts ]
                # Fill poly and make it transparant
                cv2.fillPoly(overlay, pts=[ np.array(area_pts, np.int32) ], color=area_color)

            # Not to calculate `self.area_obj_num` and `self.area_cnt`
            if in_cv_event: continue

            # Reset the object numbers in area
            self.area_obj_num[area_idx] = 0
            
            # Get the center of the area which only happend at first time
            if len(self.area_cnt)==len(self.area_pts):
                continue
            
            # Collect points data to calculate the center
            list_pnts = [ list(pts) for pts in area_pts ]
            x_max, y_max = np.max(list_pnts, axis=0)
            x_min, y_min = np.min(list_pnts, axis=0)
            
            self.area_cnt.update( {area_idx:( (x_max+x_min)//2, (y_max+y_min)//2 )} )

        return cv2.addWeighted( frame, 1-area_opacity, overlay, area_opacity, 0 )

    def cv_area_handler( self, event, x, y, flags, param):
        """ Define OpenCV Handler for Defining Area
        * param
            - type: dict
            - example: { 'img': cv_array } 
        """
        
        # print("pram type{}".format(type(param)))
        if event == cv2.EVENT_LBUTTONDOWN:
            
            # Add Index in area_points
            self.area_pts[self.area_pts_idx].append( [ x, y ] )
            self.last_frame.update({len(self.last_frame)+1:param['img'].copy()})
        
            cv2.imshow(CV_WIN, self.draw_area_event(param['img'], in_cv_event=True, draw_points=True))
        if event == cv2.EVENT_RBUTTONDOWN:
            self.area_pts[self.area_pts_idx].pop()  
             
            param['img'] = self.last_frame[len(self.last_frame)]
            del self.last_frame[len(self.last_frame)]
            
            # Display
            cv2.imshow(CV_WIN, self.draw_area_event(param['img'], in_cv_event=True, draw_points=True))
       
    def set_area(self, frame):
        """ Setup the area we want to detect via open the opencv window """        

        # Check Frame is Available
        if frame is None:

            if self.area_pts == {}: 
                msg = "Could not capture polygon coordinate and there is not provide frame to draw"
                raise Exception(msg)

            return True

        if self.area_pts != {}:
            logging.info("Detected area point is already exist")
            
        # Prepare Image to Draw
        logging.info("Detected frame, open the cv windows to collect area coordinate")

        # Initialize CV Windows
        cv2.namedWindow(CV_WIN,cv2.WINDOW_KEEPRATIO)
        # cv2.setWindowProperty(CV_WIN,cv2.WND_PROP_ASPECT_RATIO,cv2.WINDOW_KEEPRATIO)

        # Setup Mouse Event and Display
        try:
                
            while(True):
                
                # Draw Old One
                
                temp_frame = frame.copy()
                temp_frame = self.draw_area_event(temp_frame, in_cv_event=True, draw_points=True)
                
                # Init: Update Index
                self.area_pts_idx += 1  

                self.last_frame.clear()

                self.area_pts[self.area_pts_idx] = list()
                
                cv2.setMouseCallback( CV_WIN, self.cv_area_handler, {"img": temp_frame} )
                cv2.imshow(CV_WIN, self.draw_area_event(temp_frame, in_cv_event=True, draw_points=True))
                cv2.putText(temp_frame, "Click to define detected area, press any key to leave", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow(CV_WIN, temp_frame)
                key = cv2.waitKey(0)

                if key == ord('q'): break
                
        except Exception as e:
            logging.error(handle_exception(e))

        finally:
            cv2.setMouseCallback(CV_WIN, lambda *args : None)
            cv2.destroyAllWindows()

        self.check_area_pts()
        logging.info("Set up area point: {}".format(self.area_pts))

    def init_info(self):
        self.total_num  = dict()
        self.alarm      = ""
        self.app_info   = ""
        self.app_time   = get_time( False )

    def define_trigger(self, xxyy, def_trigger:dict=SENS_MAP) -> list:
        """ Define the trigger of each object
        - args
            - low       : center
            - medium    : average-4 (exclude border)
            - high      : average-9 (include border)
        """
        # verify
        for val in def_trigger.values():
            assert isinstance(val, list), "the value of def_trigger should be list "
            assert len(val)==2, "the value of def_trigger should be [ trigger_num:int, on_border:bool ]"

        # hlper function
        def get_steps(start, end, step, on_border=True): 
            return list(np.linspace( start, end, step )) if on_border \
                else list(np.linspace( start, end, step+2 ))[1:-1]

        # Start to calculate Trigger
        (x1, x2, y1, y2) = xxyy
        trigger_num, on_border = def_trigger[self.get_param(K_SENS).lower()]
        
        if trigger_num == 1:
            return [ [ (x1+x2)//2, (y1+y2)//2 ] ]
        
        x_list = get_steps( x1, x2, trigger_num, on_border )
        y_list = get_steps( y1, y2, trigger_num, on_border )

        # Return each trigger in list
        return [ [ int(x), int(y) ] for x in x_list for y in y_list ]

    def in_poly(self, pts, pts_list):
        """ Check the point is in poly area
        - args
            - pts: current point
            - pts_list: the poly area
        - return
            - type: bool
            - example: True means the point is in area
        """
        # `0`   : on the contours
        # `1`   : in the contours
        # `-1`  : out of the contours
        return cv2.pointPolygonTest(np.array(pts, np.int32), tuple(pts_list), False)!=(-1)
        
    def check_obj_area(self, xxyy:tuple):
        """ Check the object in which area, if not in area then return (-1)
        1. Get the trigger of the object
        2. If not setup area point, warning 5 times.
        3. 
        -args
            - xxyy
                - type: tuple
                - desc: the bounding box postion of the object
        - return
            - (-1)
                - descr: not in area or not set area
            - N
                - descr: in which area
        """

        # If not area setup, return (-1)
        if self.area_pts == {} or self.area_pts==None: 
            if self.frame_idx<5:
                logging.warning('No setting area point')
            return -1

        # Define trigger
        triggers = self.define_trigger( xxyy )
        
        # Check object in area or not
        for trigger in triggers:
            for idx, pts in self.area_pts.items():
                if self.in_poly(pts, trigger): 
                    return idx
        return -1

    def run(self, frame, data, draw=True) -> tuple:
        """
        Define the worflow when execute the application
        1. Capture all center point in current frame and draw the bounding box if need
        2. Calculate the distance and store in variable
        """
        if data is None:
            return frame, data

        # Init
        info = None
        self.frame_idx += 1
        self.update_draw_param( frame )

        # Draw Area
        frame = self.draw_area_event(frame)

        # Check the object in which area
        for det in data[DETS]:

            # Pass the non-interesting label    
            label = det[LABEL]
            if not (self.depend_label(label, self.get_param(K_DEPEND))): 
                continue

            # Parsing Output
            [ xmin, xmax, ymin, ymax ] = \
                [ det[key] for key in [ 'xmin', 'xmax', 'ymin', 'ymax' ] ]
            
            if self.get_param('draw_bbox'):
                color = self.get_color(label)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , self.thick)
                # Draw Text
                (t_wid, t_hei), t_base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
                t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
                cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)

                cv2.putText(
                    frame, label, (xmin, ymin-(t_base)), cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
                )

            # Counting Area Object
            cur_in_area = self.check_obj_area( (xmin, xmax, ymin, ymax) )
            if cur_in_area != (-1): 
                self.area_obj_num[cur_in_area]+=1

        # Draw Area
        for area_idx, area_cnt in self.area_cnt.items():
            
            # Area Information
            area_info = '{:03}'.format(area_idx)
            color = (0,0,255) if self.area_obj_num[area_idx]!=0 else (0,255,0)
            
            (t_wid, t_hei), t_base = cv2.getTextSize(area_info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            half_wid, half_hei = t_wid//2, t_hei//2
            cnt_x, cnt_y = area_cnt
            t_xmin, t_ymin, t_xmax, t_ymax = cnt_x-half_wid, cnt_y-half_hei, cnt_x+half_wid, cnt_y+half_hei
            
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)
            cv2.putText(
                frame, area_info, (t_xmin, t_ymin+t_hei), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )
        
        # Draw Application Information
        app_info = ', '.join( [ 'AREA-{:03} : {}'.format(key, value) for key, value in self.area_obj_num.items() ] )
        (t_wid, t_hei), t_base = cv2.getTextSize(app_info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
        t_xmin, t_ymin, t_xmax, t_ymax = 10, 10, 10+t_wid, 10+t_hei+t_base
        cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), (0,0,255) , -1)
        cv2.putText(
            frame, app_info, (t_xmin, t_ymax-t_base), cv2.FONT_HERSHEY_SIMPLEX,
            self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
        )

        return ( frame, self.area_obj_num)

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
        'draw_bbox': False,
        'draw_area': True,
        'area_opacity': 0.5    }

    ivit = get_ivit_model(model_type)
    ivit.set_async_mode()
    ivit.load_model(model_conf)
    
    # Def Application
    app = AreaDetection(
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
        elif press_key in [ord('+'), ord('=')]:
            app.set_param('area_opacity', app.get_param('area_opacity')+0.1 )
        elif press_key in [ord('-'), ord('_')]:
            app.set_param('area_opacity', app.get_param('area_opacity')-0.1 )
        
        # Calculate FPS
        if infer_data:
            fps_pool.append( int(1/(time.time()-t_start)) )
            if len(fps_pool)>10:
                fps = sum(fps_pool)//len(fps_pool)
                fps_pool.clear()

    cap.release()
    ivit.release()
