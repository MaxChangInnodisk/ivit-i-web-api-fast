import sys, os, cv2, logging, time
import numpy as np
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp

try:
    from DynamicBoundingBox import DynamicBoundingBox
    from AreaDetection import AreaDetection
except:
    from apps.DynamicBoundingBox import DynamicBoundingBox
    from apps.AreaDetection import AreaDetection

# Area
K_DEPEN         = 'depend_on'
K_AREA          = "area_points"
K_SENS          = "sensitivity"
K_DRAW_BB       = "draw_bbox"
K_DRAW_ARAE     = "draw_area"
K_AREA_COLOR    = "area_color"
K_AREA_OPACITY  = "area_opacity"
SENS_MED        = "medium"
SENS_HIGH       = "high"
SENS_LOW        = "low"
SENS_MAP = {
    SENS_LOW: [1, False],
    SENS_MED: [2, False],
    SENS_HIGH: [3, True]
}


# Counting
K_LOGIC         = 'logic'
K_LOGIC_THRES   = 'logic_thres'


class Counting(AreaDetection, ivitApp):

    def __init__(self, params=None, label=None, palette=None, log=True):
        super().__init__(params, label, palette, log)
        self.set_type('obj')
        self.init_area()
        self.init_draw_params()
        self.init_logic_param()

    def init_params(self):
        self.def_param( name=K_DEPEN, type='list', value=[], descr='add label into list if need to filter interest label' )
        self.def_param( name=K_LOGIC, type="string", value="=", descr="use logic to define situation"),
        self.def_param( name=K_LOGIC_THRES, type="int", value=3, descr="define logic threshold"),

        self.def_param( name=K_DRAW_BB, type='bool', value=True )
        self.def_param( name=K_DRAW_ARAE, type='bool', value=True )

        self.def_param( name=K_AREA_COLOR, type='list', value=(0,0,255) )
        self.def_param( name=K_AREA_OPACITY, type='float', value=0.15 )

        self.def_param( name=K_AREA, type='dict', value={} )
        self.def_param( name=K_SENS, type='str', value=SENS_LOW )

    def init_logic_param(self):
        self.operator = self.get_logic_event(self.get_param(K_LOGIC))
        self.thres = self.get_param(K_LOGIC_THRES)
        self.cur_num = {}

    def get_logic_event(self, operator):
        """ Define the logic event """
        greater = lambda x,y: x>y
        greater_or_equal = lambda x,y: x>=y
        less = lambda x,y: x<y
        less_or_equal = lambda x,y: x<=y
        equal = lambda x,y: x==y
        logic_map = {
            '>': greater,
            '>=': greater_or_equal,
            '<': less,
            '<=': less_or_equal,
            '=': equal,
        }
        return logic_map.get(operator)

    def logic_event(self, value):
        return self.operator(value)

    def run(self, frame, data, draw=True) -> tuple:
        
        # Basic in area detection
        self.frame_idx += 1
        self.update_draw_param( frame )
        frame = self.draw_area_event(frame)
        
        # Clear `self.cur_num`
        self.cur_num.clear()

        for det in (data['detections']):

            # Check Label is what we want
            if not self.depend_label(det['label'], self.get_param('depend_on')):
                continue
            
            # Parsing output
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = [ det[key] for key in [ 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax' ] ]                  

            # Check in area
            if self.check_obj_area((xmin, xmax, ymin, ymax))==(-1): 
                continue

            # Update the mount of the current object
            if self.cur_num.get(label) is None:
                self.cur_num.update( {label:0} )
            self.cur_num[label] += 1
            
            # Draw Top N label
            if not draw: continue

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.get_color(label) , self.thick)

            # Draw Text
            (t_wid, t_hei), t_base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), self.get_color(label) , -1)

            cv2.putText(
                frame, label, (xmin, ymin-(t_base)), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )


        # Draw Inforamtion
        info = ', '.join([ f'{_label}:{_num}' for _label, _num in self.cur_num.items() ])
        if info.strip() != '':
            (t_wid, t_hei), t_base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = 10, 10, 10+t_wid, 10+(t_hei+t_base)
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,255) , -1)
            cv2.putText(
                frame, info, (t_xmin, t_ymax), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )

        return ( frame, self.cur_num)

if __name__ == "__main__":

    import cv2
    from ivit_i.common.model import get_ivit_model

    # Define iVIT Model
    model_type = 'obj'
    model_anchor = [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326 ]
    model_conf = { 
        "tag": model_type,
        "openvino": {
            "model_path": "./model/yolo-v3-tf/FP32/yolo-v3-tf.xml",
            "label_path": "./model/yolo-v3-tf/coco.names",
            "anchors": model_anchor,
            "architecture_type": "yolo",
            "device": "CPU",
            "thres": 0.9
        }
    }

    ivit = get_ivit_model(model_type)
    ivit.load_model(model_conf)
    ivit.set_async_mode()
    
    # Def Application
    app_conf = {
        'depend_on': [],
        'draw_area': True,
    }
    app = Counting( 
        params=app_conf, 
        label=model_conf['openvino']['label_path']
    )

    # Get Source
    src_path = './data/car.mp4'   # 1280x720
    src_path = './data/4-corner-downtown.mp4' # 1920x1080
    cap = cv2.VideoCapture(src_path)

    # Set up Area
    ret, frame = cap.read()
    app.set_area(frame)

    output = None
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret: break
        _output = ivit.inference(frame=frame)
        output = _output if _output is not None else output 
        if (output):
            frame, info = app(frame, output)
            print(info)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break

    cap.release()
    ivit.release()
