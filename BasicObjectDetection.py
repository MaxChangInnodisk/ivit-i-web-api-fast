import sys, os, cv2
import numpy as np
sys.path.append( os.getcwd() )
from ivit_i.app.common import ivitApp
import random
import logging


class BasicObjectDetection(ivitApp):
    """ Basic Object Detection Application
    * Parameters
        1. depend_onarea_opacity
    * Function
        1. depend_label()
    """
    def __init__(self, params=None, label=None, palette=None, log=True):
        self.app_type = 'obj'
        self.params = params
        self.depend_on ={}
        self.palette={}
        self.init_draw_params()

    def init_draw_params(self):
        """ Initialize Draw Parameters """
        #for draw result and boundingbox
        self.frame_idx = 0
        self.frame_size = None
        self.font_size  = None
        self.font_thick = None
        self.thick      = None
        self.draw_result =True
        self.draw_bbox =True

        #for draw area
        self.draw_area=True
        self.area_opacity=None
        self.area_color=None
        self.area_pts={}
        self.area_cnt = {}
   
    def update_draw_param(self, frame):
        """ Update the parameters of the drawing tool, which only happend at first time. """
        
        # if frame_size not None means it was already init 
        if( self.frame_idx > 1): return None

        # Parameters
        FRAME_SCALE     = 0.0005    # Custom Value which Related with Resolution
        BASE_THICK      = 1         # Setup Basic Thick Value
        BASE_FONT_SIZE  = 0.5   # Setup Basic Font Size Value
        FONT_SCALE      = 0.2   # Custom Value which Related with the size of the font.

        # Get Frame Size
        self.frame_size = frame.shape[:2]
        
        # Calculate the common scale
        scale = FRAME_SCALE * sum(self.frame_size)
        
        # Get dynamic thick and dynamic size 
        self.thick  = BASE_THICK + round( scale )
        self.font_thick = self.thick//2
        self.font_size = BASE_FONT_SIZE + ( scale*FONT_SCALE )
        self.draw_result = self.params['application']['draw_result'] if self.params['application']['draw_result'] else True
        self.draw_bbox = self.params['application']['draw_bbox'] if self.params['application']['draw_bbox'] else True
        for i in range(len(self.params['application']['areas'])):
            if self.params['application']['areas'][i]['area_point']!=[]:
                self.area_pts.update({i:self.params['application']['areas'][i]['area_point']})
            else:
                self.area_pts.update({i:[[0,0],[frame.shape[1],0],[frame.shape[1],frame.shape[0]],[0,frame.shape[0]]]})
        self.area_color=[0,0,255]
        self.area_opacity=0.4
        logging.info('Frame: {} ({}), Get Border Thick: {}, Font Scale: {}, Font Thick: {}'
            .format(self.frame_size, scale, self.thick, self.font_size, self.font_thick))    
    
    def collect_depand_info(self):
        depand_flag={}
        for i in range(len(self.params['application']['areas'])): 
            if len(self.params['application']['areas'][i]['depend_on'])>0:
                        
                self.depend_on.update({i:self.params['application']['areas'][i]['depend_on']})
                depand_flag.update({i:True})
                
            else:
                depand_flag.update({i:False})
                
        return depand_flag    


    def custom_function(self, frame, color:tuple, label,score, left_top:tuple, right_down:tuple):
        """ The draw method customize by user 
        Do something you want to do
        """
        
        (xmin, ymin), (xmax, ymax) = left_top, right_down
        info = '{} {:.1%}'.format(label, score)
        
        # Draw bounding box
        if self.draw_bbox:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color , self.thick)

        
        # Draw Text
        if self.draw_result:
            (t_wid, t_hei), t_base = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, self.font_size, self.font_thick)
            t_xmin, t_ymin, t_xmax, t_ymax = xmin, ymin-(t_hei+(t_base*2)), xmin+t_wid, ymin
            cv2.rectangle(frame, (t_xmin, t_ymin), (t_xmax, t_ymax), color , -1)
            cv2.putText(
                frame, info, (xmin, ymin-(t_base)), cv2.FONT_HERSHEY_SIMPLEX,
                self.font_size, (255,255,255), self.font_thick, cv2.LINE_AA
            )

        return frame
    def draw_area_event(self, frame, area_color=None, area_opacity=None, in_cv_event=False, draw_points=True, draw_polys=True):
        """ Draw Detecting Area and update center point if need.
        - args
            - frame: input frame
            - area_color: control the color of the area
            - area_opacity: control the opacity of the area
        """
        
        # Get Parameters
        area_color = self.area_color if area_color is None else area_color
        area_opacity = self.area_opacity if area_opacity is None else area_opacity
        
        #FIXME: Change to bitwise and store the template at first to speed up 
        # Parse All Area
        overlay = frame.copy()
        for area_idx, area_pts in self.area_pts.items():
            
            if area_pts==[]: continue

            # Draw
            if self.draw_area:
                # Draw circle if need
                if draw_points: 
                    [ cv2.circle(frame, tuple(pts), 3, area_color, -1) for pts in area_pts ]
                # Fill poly and make it transparant
              
                cv2.fillPoly(overlay, pts=[ np.array(area_pts) ], color=area_color)

            if in_cv_event: continue


        return cv2.addWeighted( frame, 1-area_opacity, overlay, area_opacity, 0 )
    def inpolygon(self,px,py,poly):
        is_in = False
        for i , corner in enumerate(poly):
            next_i = i +1 if i +1 < len(poly) else 0
            x1 ,y1 = corner
            x2 , y2=poly[next_i]
            if (x1 == px and y1 ==py) or (x2==px and y2 ==py):
                is_in = False
                break
            if min(y1,y2) <py <= max(y1 ,y2):
                x =x1+(py-y1)*(x2-x1)/(y2-y1)
                if x ==px:
                    is_in = False
                    break
                elif x > px:
                    is_in = not is_in
        return is_in 
    def __call__(self, frame, data, draw=True) -> tuple:
        #collect depend_on for each area from config
        
        output_result={}
        temp_save_info={}
        depand_flag=self.collect_depand_info()
        self.update_draw_param(frame=frame)
        # print(len(self.area_pts))
        if len(self.area_pts)>0:
            frame = self.draw_area_event(frame)
        for id,det in enumerate(data['detections']):
            # Check Label is what we want
            ( label, score, xmin, ymin, xmax, ymax ) \
                 = [ det[key] for key in [ 'label', 'score', 'xmin', 'ymin', 'xmax', 'ymax' ] ]                  
            temp_save_info.update({'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax,'label':label,'score':score,'id':id})
            for i in range(len(depand_flag)):
                
                if self.inpolygon(((xmin+xmax)/2),((ymin+ymax)/2),self.area_pts[i]):
               #check area[i] depand is not None 
                    if depand_flag[i]:
                        if label in self.depend_on[i]:  
                            # Draw Top N label
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
        output_result.update({"results":[temp_save_info]})                        
        return ( frame, output_result)

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
    
    # Def Application
    app_config={
        "application": 
        {
            "name": "CountingArea",
            "areas": [
                    {
                            "name": "first area",
                            "depend_on": ['car'],
                            "palette": {
                                "car": [ 255, 0, 0 ],
                                "truck": [ 0, 0, 255 ]
                            },
                            "area_point": [ ], 
                    },
                    {
                            "name": "second area",
                            "depend_on": [ "cat" ,"pig"],
                            "area_point": [[0,0], [640, 0], [480, 640], [0, 480] ], 
                    }
            ],
            "draw_result": False,
            "draw_bbox":True,
        }
    }
    app = BasicObjectDetection(params=app_config,label=model_conf['openvino']['label_path'])

    # Get Source
    cap = cv2.VideoCapture('./data/4-corner-downtown.mp4')
    while(cap.isOpened()):

        ret, frame = cap.read()
        if not ret: break
        output = ivit.inference(frame=frame)

        frame, info = app(frame, output)
        
        print(info)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) in [ ord('q'), 27 ]: break

    cap.release()
    ivit.release()
