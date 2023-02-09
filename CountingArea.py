import sys, os, cv2, logging, time
import numpy as np
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp

try:
    from BasicObjectDetection import BasicObjectDetection
    
except:
    from apps.BasicObjectDetection import BasicObjectDetection
    

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


class Counting(BasicObjectDetection, ivitApp):

    def __init__(self, params=None, label=None, palette=None, log=True):
       
        self.app_type = 'obj'
        self.params = params
        self.depend_on ={}
        self.palette={}
        self.counter={}
        self.judge_area=10
        self.event_title=""
        # self.init_area()
        self.init_draw_params()
        self.init_logic_param()

    def init_logic_param(self):
        self.operator = self.get_logic_event(self.params['application']['areas'][0]['events'][0]['logic_operator'])
        self.thres = self.params['application']['areas'][0]['events'][0]['logic_value']


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

    def logic_event(self, value,thres):
        return self.operator(value,thres)
    
    def __call__(self, frame, data, draw=True) -> tuple:
        
        # Basic in area detection
        self.frame_idx += 1
        depand_flag=self.collect_depand_info()
        self.update_draw_param( frame )

        frame = self.draw_area_event(frame)
        
        # Clear `self.cur_num`
        self.counter.clear()
        self.event_title=""
        for id,det in enumerate(data['detections']):
            # Parsing output
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = [ det[key] for key in [ 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax' ] ]                  

            for i in range(len(depand_flag)):
                
                if self.inpolygon(((xmin+xmax)/2),((ymin+ymax)/2),self.area_pts[i]):
               #check area[i] depand is not None 
                    if depand_flag[i]:
                        if label in self.depend_on[i]:  
                        
                            # Draw Top N label
                            if self.counter.__contains__(label):
                                _=self.counter[label]+1
                                self.counter.update({label:_})
                            else:
                                self.counter.update({label:1})     
                            
                            if not draw: continue
                            # Further Process

                            frame = self.custom_function(
                                frame = frame,
                                color = self.params['application']['areas'][i]['palette'].get(label, [0,0,0] ),
                                label = '{} {:.1%}'.format(label, score),
                                score=score,
                                left_top = (xmin, ymin),
                                right_down = (xmax, ymax)
                            )
                    else:
                        frame = self.custom_function(
                                    frame = frame,
                                    color = self.params['application']['areas'][i]['palette'].get(label, [0,0,0] ),
                                    label = '{} {:.1%}'.format(label, score),
                                    score=score,
                                    left_top = (xmin, ymin),
                                    right_down = (xmax, ymax)
                                )    
        # print("10 {} {}".format(self.thres,self.logic_event(10,self.thres)))                        
        if self.logic_event(10,self.thres):
            self.event_title=self.params['application']['areas'][0]['events'][0]['title']
        # Draw Inforamtion
        info = self.event_title+" "+' , '.join([ f'{_label}:{_num}' for _label, _num in self.counter.items() ])
        if info.strip() != '':
            (t_wid, t_hei), t_base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = 10, 10, 10+t_wid, 10+(t_hei+t_base)
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax+t_base), (0,0,255) , -1)
            cv2.putText(
                frame, info, (t_xmin, t_ymax), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )
    
        return ( frame, self.counter)

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
        "application": {
		                "name": "CountingArea",
		                "areas": [
                        {
                                "name": "The intersection of Datong Rd",
                                "depend_on": [ "car", "truck", "motocycle" ],
                                "palette": {
                                "car": [ 255, 0, 0 ],
                                "truck": [ 0, 0, 255 ]
                            },
                                "area_point": [ [0,0], [640, 0] , [480, 640], [0, 480]], 
                                "events": [
                                        {
                                                "title": "The daily traffic is over 2.",
                                                "logic_operator": ">",
                                                "logic_value": 2,
                                        }
                                ]
                        }],
        "draw_result": False,
        "draw_bbox":True,

}
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
    # app.set_area(frame)

    output = None
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret: break
        _output = ivit.inference(frame=frame)
        output = _output if _output is not None else output 
        if (output):
            frame, info = app(frame, output)
            # print(info)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break

    cap.release()
    ivit.release()
